"""
    CellAverageFunctional

A functional that computes the cell average of a function over a control volume.

The functional is defined as:
    λ(f) = (1/|V|) ∫_V f(x) dx

where V is any parametrized geometry from Meshes.jl and |V| is its volume measure.
Integration is performed via `Meshes.integral` (h-adaptive cubature).

This functionality is provided by the KernelInterpolationCellAverageExt extension and
requires Meshes.jl to be loaded.

# Fields
- `volume::Any`: The control volume — any parametrized Meshes.jl geometry
- `volume_measure::RealT`: The volume measure |V| in the same precision as the geometry

The type parameter `RealT` is determined by the geometry's coordinate type (e.g. `Float64`
or `BigFloat`) and controls the floating-point precision used throughout quadrature.
"""
struct CellAverageFunctional{Dim, RealT <: Real}
    volume::Any          # any parametrized Meshes.jl geometry
    volume_measure::RealT
end

"""
    assemble_cell_average_matrix(functionals, kernel)

Assemble the kernel Gram matrix for cell-average interpolation. Entry ``(i,j)`` is

```math
    A_{ij} = \\frac{1}{|V_i||V_j|}\\int_{V_i}\\int_{V_j} K(x,y)\\,\\mathrm{d}y\\,\\mathrm{d}x.
```

The element type `RealT` of the returned matrix is inferred from the `RealT` parameter of
the `CellAverageFunctional`s, which is determined by the coordinate type of the underlying
geometry. Construct the functionals with BigFloat geometry (via `regular_cells(...; RealT=BigFloat)`)
and call `setprecision(BigFloat, bits)` beforehand to perform assembly and solve in higher
precision.
Requires Meshes.jl.

See also [`cell_average_interpolate`](@ref).
"""
function assemble_cell_average_matrix end

"""
    centroid_enclosing_radius(geometry)

Return the radius of the smallest ball centered at the centroid of `geometry` that
contains it, i.e. the maximum distance from the centroid to any vertex:
```math
    r(V) = \\max_{v \\in \\mathrm{vertices}(V)} \\|v - \\mathrm{centroid}(V)\\|.
```
Requires `Meshes.jl`. Works for any geometry with `centroid` and `vertices` defined
(e.g. `Polytope`, `Segment`). A vector of geometries returns the maximum radius.

See also [`maximum_cell_diameter`](@ref).
"""
function centroid_enclosing_radius end

"""
    centroid_nodeset(functionals)

Return a [`NodeSet`](@ref) containing the centroids of all control volumes in
`functionals`. Requires Meshes.jl.

See also [`separation_distance`](@ref), [`maximum_cell_diameter`](@ref).
"""
function centroid_nodeset end
