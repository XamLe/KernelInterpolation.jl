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
- `volume_measure::Float64`: The volume measure |V|
"""
struct CellAverageFunctional{Dim}
    volume::Any          # any parametrized Meshes.jl geometry
    volume_measure::Float64
end

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
