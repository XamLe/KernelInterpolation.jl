module KernelInterpolationCellAverageExt

using LinearAlgebra: Symmetric
using Meshes: Meshes, Point, measure, integral, to, ustrip

if pkgversion(Meshes) < v"0.57"
    error("""
    KernelInterpolationCellAverageExt requires Meshes ≥ 0.57.
    Please upgrade: ] update Meshes
    """)
end

using KernelInterpolation: KernelInterpolation

# Helper function to strip units from Meshes.jl Points
function _to_coords(p::Point)
    return ustrip.(to(p))
end

# Override the constructor to accept any Meshes geometry
# Using the abstract Geometry type to capture all geometries
function KernelInterpolation.CellAverageFunctional(volume)
    vol_measure = ustrip(measure(volume))
    return KernelInterpolation.CellAverageFunctional{Meshes.embeddim(volume)}(volume, vol_measure)
end

"""
    assemble_cell_average_matrix(functionals::Vector{CellAverageFunctional}, 
                                 kernel::AbstractKernel;
                                 ibackend=nothing, 
                                 dbackend=nothing)

Assemble the kernel matrix for cell-average functionals.

For cell-average functionals λᵢ and λⱼ, the matrix entry is:
    A_ij = (1/|Vᵢ||Vⱼ|) ∫_Vᵢ ∫_Vⱼ K(x, y) dy dx

The double integral is computed using Meshes.jl's integral function via nested quadrature.
"""
function KernelInterpolation.assemble_cell_average_matrix(
    functionals::Vector{<:KernelInterpolation.CellAverageFunctional},
    kernel::KernelInterpolation.AbstractKernel;
    ibackend=nothing,
    dbackend=nothing)
    
    n = length(functionals)
    A = Matrix{Float64}(undef, n, n)
    
    for i in 1:n
        for j in 1:n
            A[i, j] = _compute_cell_average_kernel_entry(
                functionals[i], functionals[j], kernel, ibackend, dbackend
            )
        end
    end
    
    return A
end

function _compute_cell_average_kernel_entry(
    func_i::KernelInterpolation.CellAverageFunctional,
    func_j::KernelInterpolation.CellAverageFunctional,
    kernel::KernelInterpolation.AbstractKernel,
    ibackend, dbackend)
    
    V_i = func_i.volume
    V_j = func_j.volume
    measure_i = func_i.volume_measure
    measure_j = func_j.volume_measure
    
    # Inner integral: ∫_Vⱼ K(x, y) dy for a fixed x
    function inner_integral_wrapper(x)
        x_vec = _to_coords(x)

        function integrand_inner(y)
            y_vec = _to_coords(y)
            return kernel(x_vec, y_vec)
        end
        
        if ibackend !== nothing && dbackend !== nothing
            return integral(integrand_inner, V_j; ibackend=ibackend, dbackend=dbackend)
        else
            return integral(integrand_inner, V_j)
        end
    end
    
    # Outer integral: ∫_Vᵢ (∫_Vⱼ K(x, y) dy) dx
    if ibackend !== nothing && dbackend !== nothing
        result = integral(inner_integral_wrapper, V_i; ibackend=ibackend, dbackend=dbackend)
    else
        result = integral(inner_integral_wrapper, V_i)
    end

    # Normalize by both volume measures (ustrip removes Unitful units from the integral result)
    return ustrip(result) / (measure_i * measure_j)
end

function KernelInterpolation.cell_average_interpolate(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional},
        values::AbstractVector{<:Real},
        kernel::KernelInterpolation.AbstractKernel;
        ibackend = nothing,
        dbackend = nothing,
        linsolve = nothing)
    n = length(functionals)
    @assert length(values) == n "number of values must match number of functionals"

    A = KernelInterpolation.assemble_cell_average_matrix(functionals, kernel;
                                                          ibackend = ibackend,
                                                          dbackend = dbackend)
    A_sym = Symmetric(A)
    c = KernelInterpolation.solve_linear_system(A_sym, Float64.(values), linsolve)

    return KernelInterpolation.CellAverageInterpolation(kernel, functionals, c, A_sym)
end

function (itp::KernelInterpolation.CellAverageInterpolation)(x::AbstractVector)
    s = zero(eltype(itp.c))
    for (j, func) in enumerate(itp.functionals)
        # λⱼ(K(x, ·)) = (1/|Vⱼ|) ∫_Vⱼ K(x, y) dy
        integrand = y -> itp.kernel(x, _to_coords(y))
        s += itp.c[j] * ustrip(integral(integrand, func.volume)) / func.volume_measure
    end
    return s
end

function (itp::KernelInterpolation.CellAverageInterpolation{1})(x::Real)
    return itp([x])
end

function KernelInterpolation.cell_averages(itp::KernelInterpolation.CellAverageInterpolation)
    result = Vector{Float64}(undef, length(itp.functionals))
    for (i, func) in enumerate(itp.functionals)
        integrand = p -> itp(_to_coords(p))
        result[i] = ustrip(integral(integrand, func.volume)) / func.volume_measure
    end
    return result
end

end
