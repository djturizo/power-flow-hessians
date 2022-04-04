include("power_flow_rect_qc.jl")
import .PowerFlowRectQC as QC


function main()
    case_dir = "MATPOWER/pglib_opf_case118_ieee.m"
    case_dir = "MATPOWER/pglib_opf_case2383wp_k.m"
    QC.compute_qc(case_dir, true)
end


main();
