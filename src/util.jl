"""
    examples_dir()

Return the directory where the example files provided with KernelInterpolation.jl are located.

# Examples
```@example
readdir(examples_dir())
```
"""
examples_dir() = pkgdir(KernelInterpolation, "examples")::String

"""
    get_examples()

Return a list of all examples that are provided by KernelInterpolation.jl. See also
[`examples_dir`](@ref) and [`default_example`](@ref).
"""
function get_examples()
    examples = String[]
    for (root, dirs, files) in walkdir(examples_dir())
        for f in files
            if endswith(f, ".jl")
                push!(examples, joinpath(root, f))
            end
        end
    end

    return examples
end

"""
    default_example()

Return the path to an example that can be used to quickly see KernelInterpolation.jl in action.
See also [`examples_dir`](@ref) and [`get_examples`](@ref).
"""
function default_example()
    return joinpath(examples_dir(), "interpolation", "interpolation_2d.jl")
end

# Create `d` polyvars from `TypedPolynomials.jl`, don't use `@polyvars` because of
# https://github.com/JuliaAlgebra/TypedPolynomials.jl/issues/51, instead use the
# workaround from there
"""
    l2_error(itp, f, nodes; domain_vol)

Compute the discrete L₂ error of `itp` against reference function `f` evaluated
at `nodes` (any iterable of point vectors). `domain_vol` is the measure of the
integration domain; for a domain [a,b]^d pass `domain_vol = (b-a)^d`. With a
uniform grid of N points the approximation is √(domain_vol/N) · ‖e‖₂.

!!! warning "Uniform grid assumption"
    This function assumes equal quadrature weight `domain_vol/N` per point.
    For non-uniform node distributions (Chebyshev nodes, adaptive grids, random
    scatter) the weights differ between points and this estimate will be biased.
    In that case compute the error with explicit per-point weights instead.

See also [`linf_error`](@ref).
"""
function l2_error(itp, f, nodes; domain_vol)
    e = [itp(x) - f(x) for x in nodes]
    return sqrt(domain_vol / length(e)) * norm(e)
end

"""
    linf_error(itp, f, nodes)

Compute the discrete L₋∞ error of `itp` against reference function `f` evaluated
at `nodes` (any iterable of point vectors).

See also [`l2_error`](@ref).
"""
function linf_error(itp, f, nodes)
    return maximum(abs(itp(x) - f(x)) for x in nodes)
end

"""
    l2_error(itp_vals, f_vals; domain_vol)

Compute the discrete L₂ error from precomputed vectors of interpolant and reference values.
`domain_vol` is the measure of the integration domain. See [`l2_error(itp, f, nodes)`](@ref)
for the uniform-grid assumption and its limitations.
"""
function l2_error(itp_vals::AbstractVector, f_vals::AbstractVector; domain_vol)
    return sqrt(domain_vol / length(itp_vals)) * norm(itp_vals .- f_vals)
end

"""
    linf_error(itp_vals, f_vals)

Compute the discrete L₋∞ error from precomputed vectors of interpolant and reference values.
"""
function linf_error(itp_vals::AbstractVector, f_vals::AbstractVector)
    return maximum(abs, itp_vals .- f_vals)
end

polyvars(d) = ntuple(i -> Variable{Symbol("x[", i, "]")}(), d)
# The function above is not type stable.
# Therefore, we define some common special cases for performance reasons.
polyvars(::Val{1}) = (Variable{Symbol("x[1]")}(),)
polyvars(::Val{2}) = (Variable{Symbol("x[1]")}(), Variable{Symbol("x[2]")}())
function polyvars(::Val{3})
    return (Variable{Symbol("x[1]")}(), Variable{Symbol("x[2]")}(),
            Variable{Symbol("x[3]")}())
end
