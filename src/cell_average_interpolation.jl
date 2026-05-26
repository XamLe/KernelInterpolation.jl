@doc raw"""
    CellAverageInterpolation

Interpolation object for cell-average data. Represents the function
```math
    s(x) = \sum_{j=1}^N c_j\lambda_j(K(x, \cdot))
         = \sum_{j=1}^N \frac{c_j}{|V_j|}\int_{V_j} K(x, y)\,\mathrm{d}y,
```
where ``\lambda_j`` are [`CellAverageFunctional`](@ref)s and the coefficients ``c_j``
solve the linear system ``Ac = \bar{f}`` with matrix entries
```math
    A_{ij} = \frac{1}{|V_i||V_j|}\int_{V_i}\int_{V_j} K(x,y)\,\mathrm{d}y\,\mathrm{d}x.
```

Constructed via [`cell_average_interpolate`](@ref). Requires Meshes.jl.

# Fields
- `kernel`: the kernel function
- `functionals::Vector{CellAverageFunctional}`: the cell-average functionals
- `c::Vector{RealT}`: the coefficient vector
- `system_matrix`: the assembled (and possibly factorized) matrix ``A``

See also [`CellAverageFunctional`](@ref), [`cell_average_interpolate`](@ref).
"""
struct CellAverageInterpolation{Dim, RealT, KernelT, A}
    kernel::KernelT
    functionals::Vector{CellAverageFunctional{Dim}}
    c::Vector{RealT}
    system_matrix::A
end

function Base.show(io::IO, itp::CellAverageInterpolation)
    n = length(itp.functionals)
    return print(io,
                 "CellAverageInterpolation with $n functionals and kernel $(itp.kernel).")
end

"""
    interpolation_kernel(itp::CellAverageInterpolation)

Return the kernel of a [`CellAverageInterpolation`](@ref).
"""
interpolation_kernel(itp::CellAverageInterpolation) = itp.kernel

"""
    coefficients(itp::CellAverageInterpolation)

Return the coefficient vector of a [`CellAverageInterpolation`](@ref).
"""
coefficients(itp::CellAverageInterpolation) = itp.c

"""
    system_matrix(itp::CellAverageInterpolation)

Return the system matrix of a [`CellAverageInterpolation`](@ref).
"""
system_matrix(itp::CellAverageInterpolation) = itp.system_matrix

"""
    dim(itp::CellAverageInterpolation)

Return the spatial dimension of a [`CellAverageInterpolation`](@ref).
"""
dim(::CellAverageInterpolation{Dim}) where {Dim} = Dim

"""
    functionals(itp::CellAverageInterpolation)

Return the vector of [`CellAverageFunctional`](@ref)s of a
[`CellAverageInterpolation`](@ref).
"""
functionals(itp::CellAverageInterpolation) = itp.functionals

@doc raw"""
    cell_average_interpolate(functionals, values, kernel;
                             ibackend = nothing, dbackend = nothing, linsolve = nothing)

Interpolate cell-average `values` using the kernel `kernel` and the
[`CellAverageFunctional`](@ref)s in `functionals`. Determines the coefficients ``c_j``
in
```math
    s(x) = \sum_{j=1}^N c_j\lambda_j(K(x, \cdot))
```
by solving the linear system ``Ac = \bar{f}`` with matrix entries
```math
    A_{ij} = \frac{1}{|V_i||V_j|}\int_{V_i}\int_{V_j} K(x,y)\,\mathrm{d}y\,\mathrm{d}x.
```
Returns a [`CellAverageInterpolation`](@ref) that can be evaluated at any point.

Requires Meshes.jl. The optional `ibackend` and `dbackend` keyword arguments are
forwarded to Meshes.jl's `integral` for the quadrature backend. If `linsolve` is
provided it is passed to LinearSolve.jl; otherwise the backslash operator is used.

See also [`CellAverageFunctional`](@ref), [`assemble_cell_average_matrix`](@ref).
"""
function cell_average_interpolate end

"""
    cell_averages(itp::CellAverageInterpolation)

Numerically compute the cell averages ``\\lambda_i(s)`` of the interpolant `itp`
over each of its control volumes. Requires Meshes.jl.
"""
function cell_averages end
