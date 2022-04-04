module PowerFlowRectQC

import PowerModels as PM, SparseArrays as SA, ForwardDiff as FD
import SparseDiffTools as SDT, SparsityDetection as SD
import SymbolicAD as SAD


macro fullprint(var)
    return esc(:(show(stdout, "text/plain", $var); println()))
end


function parse_case(case_dir::String)
    # Load test case
    mpc = PM.make_basic_network(PM.parse_file(case_dir))
    n = length(mpc["bus"])
    g = length(mpc["gen"])
    Y = PM.calc_basic_admittance_matrix(mpc)
    G = real.(Y)
    B = imag.(Y)
    Yrec = vcat(hcat(G, -B), hcat(B, G))

    # Voltage phasors
    result = PM.compute_ac_pf(mpc)
    if result["termination_status"]
        # Power flow converged
        V0 = zeros(ComplexF64, n)
        for (key, bus) in result["solution"]["bus"]
            V0[parse(Int64, key)] = bus["vm"] * 
                exp(im*bus["va"])
        end
    else
        println("Power flow did not converge! " *
            "Using flat start as the operating point")
        V0 = ones(ComplexF64, n)
    end
    x0 = vcat(real.(V0), imag.(V0))

    # Generator variables
    Vg = zeros(Float64, n)
    Pg = zeros(Float64, n)
    # Power demands
    Pd = zeros(Float64, n)
    Qd = zeros(Float64, n)

    # Logical indices of PQ, PV and slack buses
    c_pv = fill(false, n)
    c_vd = fill(false, n)
    c_pq = fill(false, n)
    c_pvq = fill(false, n) # PV and PQ nodes
    c_pvd = fill(false, n) # PV and slack nodes

    # Parse inputs from "bus" dict
    for (_, bus) in mpc["bus"]
        key = bus["bus_i"]
        if bus["bus_type"] == 3
            # Slack bus
            c_vd[key] = true
            c_pvd[key] = true
        else
            c_pvq[key] = true
            if bus["bus_type"] == 2
                # PV bus
                c_pv[key] = true
                c_pvd[key] = true
            else
                # PQ bus
                c_pq[key] = true
            end
        end
    end

    # Parse inputs from "gen" dict
    # Multiple generators may be connected to the same bus,
    # the power parameters are added to represent a single
    # machine per bus.
    for (_, gen) in mpc["gen"]
        key = gen["gen_bus"]
        Pg[key] += gen["pg"]
        # The voltage for this bus will be the one of the last
        # parsed gen.
        Vg[key] = gen["vg"]
    end

    # Parse inputs from "load" dict
    # Just in case, we also add load powers
    for (_, load) in mpc["load"]
        key = load["load_bus"]
        Pd[key] += load["pd"]
        Qd[key] += load["qd"]
    end

    # Declare type of output variables and return them
    case = @NamedTuple{n::Int64, g::Int64, 
        Yrec::SA.SparseMatrixCSC{Float64,Int64}, V0::Vector{ComplexF64},
        x0::Vector{Float64}}(
        (n, g, Yrec, V0, x0))
    choosers = @NamedTuple{vd::Vector{Bool}, pv::Vector{Bool},
        pq::Vector{Bool}, pvq::Vector{Bool}, pvd::Vector{Bool}}(
        (c_vd, c_pv, c_pq, c_pvq, c_pvd))
    input = @NamedTuple{Vg::Vector{Float64}, Pg::Vector{Float64},
        Pd::Vector{Float64}, Qd::Vector{Float64}}(
        (Vg[c_pvd], Pg[c_pvd], Pd, Qd))
    return case, choosers, input
end


function pf_hessian(case, ind)::Tuple{
    Vector{SA.SparseMatrixCSC{Float64,Int64}},
    Vector{SA.SparseMatrixCSC{Float64,Int64}},
    Vector{SA.SparseMatrixCSC{Float64,Int64}},
    Vector{SA.SparseMatrixCSC{Float64,Int64}},
    SA.SparseMatrixCSC{Float64,Int64}}

    n = case.n
    Yrec = case.Yrec
    # Power flow Hessian
    Hpf = fill(SA.spzeros(Float64, 2*n, 2*n), 2*n)
    # Active power Hessians
    HP = fill(SA.spzeros(Float64, 2*n, 2*n), n)
    # Reactive power Hessians
    HQ = fill(SA.spzeros(Float64, 2*n, 2*n), n)
    # Squared voltage Hessians
    HV = fill(SA.spzeros(Float64, 2*n, 2*n), n)
    # Total loss Hessian
    Hloss = SA.spzeros(Float64, 2*n, 2*n)
    for k in 1:n
        # Compute P Hessians
        HP[k] = SA.sparse([k], [1], [1.0], 2*n, 1) * Yrec[k:k,:]
        HP[k] += SA.sparse([n+k], [1], [1.0], 2*n, 1) * 
            Yrec[(n+k):(n+k),:]
        HP[k] += transpose(HP[k])
        Hloss += HP[k] # Loss Hessian is the sum of all P Hessians
        # Compute Q Hessians
        HQ[k] = -SA.sparse([k], [1], [1.0], 2*n, 1) * 
            Yrec[(n+k):(n+k),:]
        HQ[k] += SA.sparse([n+k], [1], [1.0], 2*n, 1) * 
            Yrec[k:k,:]
        HQ[k] += transpose(HQ[k])
        # Compute V² Hessians
        HV[k] = SA.sparse([k, n+k], [k, n+k], 
            [2.0, 2.0], 2*n, 2*n)
    end
    # Slack bus Hessian is 0, we don't have to compute it
    Hpf[ind.pvq] = HP[ind.pvq] # P equations
    Hpf[.+(n,ind.pq)] = HQ[ind.pq] # Q equations
    Hpf[.+(n,ind.pv)] = HV[ind.pv] # V² equations
    return Hpf, HP, HQ, HV, Hloss
end


function pf_jacobian(x::Vector{Float64}, ind_vd::Int64,
    Hpf::Vector{SA.SparseMatrixCSC{Float64,Int64}}
    )::SA.SparseMatrixCSC{Float64,Int64}
    # Compute the power flow Jacobian from the Hessians
    # For simplicity, we compute the transposed Jacobian
    n = div(length(x), 2)
    x_sp = SA.sparse(x)
    nzval = Float64[]
    rowval = Int64[]
    colptr = ones(Int64, 2*n+1)
    # Build variables for direct constructor call
    for i in 1:(2*n)
        if i<=n && i == ind_vd
            # Slack bus, Hessian is zero 
            # but Jacobian column is not
            col_i = SA.sparsevec([n+i], [1.0], 2*n)
        elseif (i-n) == ind_vd
            # Slack bus, Hessian is zero 
            # but Jacobian column is not
            col_i = SA.sparsevec([i-n], [1.0], 2*n)
        else
            # For any other bus, just get the
            # Jacobian column from the Hessian
            col_i = Hpf[i]*x_sp
        end
        nzval = vcat(nzval, col_i.nzval)
        rowval = vcat(rowval, col_i.nzind)
        colptr[i+1] = colptr[i] + length(col_i.nzval)
    end
    # Build transposed Jacobian
    Jt = SA.SparseMatrixCSC{Float64,Int64}(2*n, 2*n, 
        colptr, rowval, nzval)
    return transpose(Jt)
end


# Power flow equations
function pf(x::Vector{Float64}, case, ind, input)::Vector{Float64}
    n = case.n
    xr = x[1:n]
    xi = x[(n+1):(2*n)]
    ic = case.Yrec * x
    ir = ic[1:n]
    ii = ic[(n+1):(2*n)]
    dP = xr .* ir + xi .* ii + input.Pd
    dQ = xi .* ir - xr .* ii + input.Qd
    fx1 = zeros(Float64, n)
    fx2 = zeros(Float64, n)
    # Active power equations for PV and slack buses
    fx1[ind.pvd] = dP[ind.pvd] - input.Pg
    # Slack bus imag part equation
    fx1[ind.vd] = xi[ind.vd]
    # Active power equations for PQ nodes
    fx1[ind.pq] = dP[ind.pq]
    # Reactive power equations for PQ buses
    fx2[ind.pq] = dQ[ind.pq]
    # Squared voltage magnitude equation for slack and PV buses
    fx2[ind.pvd] = xr[ind.pvd] .^ 2 + xi[ind.pvd] .^ 2 - 
        input.Vg .^ 2
    # Slack bus real part equation
    fx2[ind.vd] = xr[ind.vd] - input.Vg[ind.gen_vd]
    return vcat(fx1, fx2)
end


# i-th scalar power flow equation
function pf_i(x::AbstractVector{T}, i::Int64, case, ind, 
    u)::T where {T<:Number}

    n = case.n;
    k = i % n
    k = (k == 0 ? n : k)::Int64
    xr = x[k]
    xi = x[n+k]
    ir = (case.Yrec[k:k,:] * x)[1]
    ii = (case.Yrec[(n+k):(n+k),:] * x)[1]
    if i <= n
        if k == ind.vd
            # Slack bus imag part equation
            f_i = xi
        else
            # Active power equations for PV or PQ bus
            f_i = xr .* ir + xi .* ii
        end
    else
        if k == ind.vd
            # Slack bus real part equation
            f_i = xr
        else
            if insorted(k, ind.pq)
                # Reactive power equations for PQ bus
                f_i = xi .* ir - xr .* ii
            else
                # Squared V² equation for PV bus
                f_i = xr^2 + xi^2
            end
        end
    end
    return f_i - u[i]
end


function compare_derivatives(case, ind, input)
    n = case.n

    # Compute Jacobian and Hessians by formula
    Hpf, = pf_hessian(case, ind)
    J0 = pf_jacobian(case.x0, ind.vd, Hpf)
    println("Hessian benchmark (formula):")
    @time pf_hessian(case, ind)
    println("Jacobian benchmark (formula):")
    @time pf_jacobian(case.x0, ind.vd, Hpf)
    println("(NonZeros, Entries) = " * 
        "($(length(J0.nzval)), $(length(case.x0)^2))")
    println("(MaxColor, Size)    = " * 
        "($(max(SDT.matrix_colors(J0)...))," * 
        " $(length(case.x0)))")
    
    # Vector of control variables
    u = -vcat(input.Pd, input.Qd)
    u[ind.pvd] += input.Pg
    u[ind.vd] = 0.0
    u[.+(n,ind.pvd)] = input.Vg .^ 2
    u[.+(n,ind.vd)] = input.Vg[ind.gen_vd]

    # Wrapper for the full power flow equations
    pfw = (x) -> map((i) -> pf_i(x, i, case, ind, u), 1:(2*n))

    # # Compute Jacobian and Hessians by automatic differentiation
    # J0_ad = FD.jacobian(pfw, case.x0)
    # Hf_i = (i) -> FD.hessian(
    #     (x) -> pf_i(x, i, case, ind, u), case.x0)
    # Hf_ad = map(Hf_i, 1:(2*n))
    # println("Hessian benchmark (ForwardDiff):")
    # @time map(Hf_i, 1:(2*n))
    # println("Jacobian benchmark (ForwardDiff):")
    # @time FD.jacobian(pfw, case.x0)
    # println("Jacobian benchmark (SparseDiffTools):")
    # SDT.forwarddiff_color_jacobian(pfw, case.x0,
    #     colorvec=SDT.matrix_colors(J0))
    # @time SDT.forwarddiff_color_jacobian(pfw, case.x0,
    # colorvec=SDT.matrix_colors(J0))

    # Compare to verify correctness
    print("PF equations max-error: ")
    @fullprint max(abs.(pfw(case.x0))...)
    print("PFv2 eqs. max-error:    ")
    @fullprint max(abs.(pf(case.x0, case, ind, input))...)
    # print("PF Jacobian max-error:  ")
    # @fullprint max(abs.(J0_ad - J0)...)
    # print("PF Hessian max-error:   ")
    # @fullprint max([max(abs.(Hf_ad[i] - Hpf[i])...) 
    #     for i in 1:(2*n)]...)
end


function compute_qc(case_dir::String, 
    check_derivatives::Bool = false)
    # Parse MATPOWER case
    case, c_old, input = parse_case(case_dir)
    n = case.n

    # SymbolicAD benchamrk
    pm = PM.instantiate_model(
        case_dir, PM.ACPPowerModel, PM.build_opf)
    SAD._nlp_block_data(pm.model; backend = SAD.DefaultBackend())
    println("Full benchmark (SymbolicAD):")
    @time SAD._nlp_block_data(pm.model; 
        backend = SAD.DefaultBackend())

    # Keep only 1 slack bus (Vd bus), make the rest PV
    ind_vd = findall(c_old.vd)[1]
    c_vd = fill(false, n)
    c_vd[ind_vd] = true
    c_pv = deepcopy(c_old.pvd)
    c_pv[ind_vd] = false
    c_pvq = collect(c_old.pq .| c_pv)
    ind_gen_vd = findfirst(findall(c_old.pvd) .== ind_vd)
    c = @NamedTuple{vd::Vector{Bool}, pv::Vector{Bool},
        pq::Vector{Bool}, pvq::Vector{Bool}, pvd::Vector{Bool}}(
        (c_vd, c_pv, c_old.pq, c_pvq, c_old.pvd))
    # TODO: compute active power of slack->PV buses

    # Cartesian indices
    ind = @NamedTuple{vd::Int64, pv::Vector{Int64},
        pq::Vector{Int64}, pvq::Vector{Int64}, 
        pvd::Vector{Int64}, gen_vd::Int64}(
        (ind_vd, findall(c.pv), findall(c.pq), 
        findall(c.pvq), findall(c.pvd), ind_gen_vd))
    
    # Compute PF Hessians and Jacobian
    Hpf, HP, HQ, HV, Hloss = pf_hessian(case,ind)
    J0 = pf_jacobian(case.x0, ind.vd, Hpf)

    # Compare derivatives if required
    if check_derivatives
        compare_derivatives(case, ind, input)
    end

    # Quadratic constraints data to be returned
    qc_data = @NamedTuple{n::Int64, x0::Vector{Float64}, 
        J0::SA.SparseMatrixCSC{Float64,Int64},
        Hpf::Vector{SA.SparseMatrixCSC{Float64,Int64}},
        Hloss::SA.SparseMatrixCSC{Float64,Int64}}(
        (n, case.x0, J0, Hpf, Hloss))
    return qc_data, ind, c
end

end
