# Regenerates ext/wendland_rectangle_integral.jl from the symbolic derivation. Not part
# of the package build or CI -- SymPy.jl is a derivation-time-only tool, deliberately not
# a package dependency (the generated file is plain Julia, no runtime Python/PyCall).
# Needs SymPy.jl in whatever environment runs this (not in the package's own Project.toml):
#     julia --project=@v1.10 -e 'import Pkg; Pkg.add("SymPy")'
#     julia --project=@v1.10 gen/derive_wendland_rectangle_integral.jl
# Full derivation, numerical validation (against dblquad/QuadGK), and the reasoning behind
# every guard in here (near-axis Taylor correction, the b1/b2-tiny whole-rectangle guard)
# is worked through in the companion notebook, analytic_associated_wendland_julia.ipynb.
using SymPy

@syms a1::(real,positive) b1::(real,positive) a2::(real,positive) b2::(real,positive) r::(real,positive) theta::(real,positive)

const NEAR_AXIS_TOL = 0.01

# phi_{k,d=2}(r), matching KernelInterpolation.jl's phi(::WendlandKernel, r) exactly.
function wendland_phi(k::Int, d::Int)
    l = fld(d, 2) + k + 1
    if k == 0
        return (1 - r)^l
    elseif k == 1
        return (1 - r)^(l + 1) * ((l + 1) * r + 1)
    elseif k == 2
        return Sym(1 // 3) * (1 - r)^(l + 2) * ((l^2 + 4l + 3) * r^2 + (3l + 6) * r + 3)
    elseif k == 3
        return Sym(1 // 15) * (1 - r)^(l + 3) * ((l^3 + 9l^2 + 23l + 15) * r^3 +
               (6l^2 + 36l + 45) * r^2 + (15l + 45) * r + 15)
    else
        throw(ArgumentError("k must be 0, 1, 2, or 3"))
    end
end

# Derives the 4 branches for a given phi and returns their SymPy expressions.
function derive_branches(phi_expr)
    H_expr = integrate(phi_expr * r, r)
    Hf_(x) = subs(H_expr, r, x)
    G_expr = integrate(phi_expr, r)
    Gf_(x) = subs(G_expr, r, x)
    Gclip_(x) = Gf_(sympy.Min(x, 1))
    H1_ = Hf_(Sym(1))

    theta_min_ = atan(a2 / b1); theta_max_ = atan(b2 / a1)
    theta_l_   = atan(a2 / a1); theta_u_   = atan(b2 / b1)

    function realify_(ex)
        ex = subs(ex, log(sin(theta) - 1), log(1 - sin(theta)))
        ex = subs(ex, log(cos(theta) - 1), log(1 - cos(theta)))
        return ex
    end
    Phi1_ = realify_(integrate(Hf_(b1 / cos(theta)), theta))
    Phi2_ = realify_(integrate(Hf_(b2 / sin(theta)), theta))
    Phi3_ = realify_(integrate(Hf_(a2 / sin(theta)), theta))
    Phi4_ = realify_(integrate(Hf_(a1 / cos(theta)), theta))
    ev_(Phi, val) = subs(Phi, theta, val)

    lo1_of_(tmin) = sympy.Min(sympy.Max(tmin, acos(sympy.Min(b1, 1))), theta_u_)
    hi2_of_(tmax) = sympy.Max(sympy.Min(tmax, asin(sympy.Min(b2, 1))), theta_u_)
    function max_side_(tmin, tmax)
        lo1_, hi2_ = lo1_of_(tmin), hi2_of_(tmax)
        (ev_(Phi1_, theta_u_) - ev_(Phi1_, tmin)) + (ev_(Phi2_, tmax) - ev_(Phi2_, theta_u_)) -
        ((ev_(Phi1_, theta_u_) - ev_(Phi1_, lo1_)) - H1_ * (theta_u_ - lo1_)) -
        ((ev_(Phi2_, hi2_) - ev_(Phi2_, theta_u_)) - H1_ * (hi2_ - theta_u_))
    end

    hi3_generic_ = sympy.Max(sympy.Min(theta_l_, asin(sympy.Min(a2, 1))), theta_min_)
    lo4_generic_ = sympy.Min(sympy.Max(theta_l_, acos(sympy.Min(a1, 1))), theta_max_)
    F_generic_ = max_side_(theta_min_, theta_max_) -
                 (ev_(Phi3_, theta_l_) - ev_(Phi3_, theta_min_)) -
                 (ev_(Phi4_, theta_max_) - ev_(Phi4_, theta_l_)) +
                 ((ev_(Phi3_, hi3_generic_) - ev_(Phi3_, theta_min_)) - H1_ * (hi3_generic_ - theta_min_)) +
                 ((ev_(Phi4_, theta_max_) - ev_(Phi4_, lo4_generic_)) - H1_ * (theta_max_ - lo4_generic_))

    theta_max_a2_0_ = atan(b2 / a1)
    lo4_a2_0_ = sympy.Min(sympy.Max(Sym(0), acos(sympy.Min(a1, 1))), theta_max_a2_0_)
    F_a2_0_ = max_side_(Sym(0), theta_max_a2_0_) -
              (ev_(Phi4_, theta_max_a2_0_) - ev_(Phi4_, Sym(0))) +
              ((ev_(Phi4_, theta_max_a2_0_) - ev_(Phi4_, lo4_a2_0_)) - H1_ * (theta_max_a2_0_ - lo4_a2_0_)) -
              a2 * (Gclip_(b1) - Gclip_(a1))

    theta_min_a1_0_ = atan(a2 / b1)
    hi3_a1_0_ = sympy.Max(sympy.Min(sympy.pi / 2, asin(sympy.Min(a2, 1))), theta_min_a1_0_)
    F_a1_0_ = max_side_(theta_min_a1_0_, sympy.pi / 2) -
              (ev_(Phi3_, sympy.pi / 2) - ev_(Phi3_, theta_min_a1_0_)) +
              ((ev_(Phi3_, hi3_a1_0_) - ev_(Phi3_, theta_min_a1_0_)) - H1_ * (hi3_a1_0_ - theta_min_a1_0_)) -
              a1 * (Gclip_(b2) - Gclip_(a2))

    F_both_0_ = max_side_(Sym(0), sympy.pi / 2) -
                a1 * (Gclip_(b2) - Gclip_(Sym(0))) - a2 * (Gclip_(b1) - Gclip_(Sym(0)))

    for (nm, ex) in [("generic", F_generic_), ("a2_0", F_a2_0_), ("a1_0", F_a1_0_), ("both_0", F_both_0_)]
        s = string(ex)
        @assert !occursin("Piecewise", s) && !occursin("Integral", s) && !occursin(r"\bI\b", s) && !occursin("nan", s) nm
    end

    # Gclip itself, compiled: needed separately (not just as a building block for the
    # branches above) for the b1-tiny/b2-tiny guard in the dispatcher below.
    Gclip_sym = Gclip_(Sym(:y))

    return (generic = F_generic_, a2_0 = F_a2_0_, a1_0 = F_a1_0_, both_0 = F_both_0_,
            Gclip = Gclip_sym)
end

# Extracts the lambdify-generated function body as Julia source text.
function body_source(ex, vars)
    lam = lambdify(ex, vars; expression = Val{true})
    return string(lam.args[2])
end

out = IOBuffer()
println(out, "# AUTO-GENERATED by gen/derive_wendland_rectangle_integral.jl -- do not edit by hand.")
println(out, "# Regenerate: julia --project=<env with SymPy> gen/derive_wendland_rectangle_integral.jl")
println(out, "# Full derivation and validation: analytic_associated_wendland_julia.ipynb")
println(out, "#")
println(out, "# Closed-form double integral of the Wendland kernel polynomial phi_{k,d=2} over an")
println(out, "# axis-aligned rectangle [a1,b1] x [a2,b2] in the open first quadrant (0 <= a1 < b1,")
println(out, "# 0 <= a2 < b2), split into four branches handling the axis-touching boundary cases")
println(out, "# (a1=0 and/or a2=0) separately -- evaluating the generic formula there hits a genuine")
println(out, "# catastrophic-cancellation singularity, not just reduced accuracy. Each boundary")
println(out, "# branch carries a linear Taylor correction giving O(tol^2) accuracy instead of O(tol).")
println(out, "# A second guard handles the whole rectangle (not just one edge) collapsing toward an")
println(out, "# axis -- e.g. a query point an infinitesimal distance inside a cell edge, via the")
println(out, "# quadrant-splitting wrapper below -- which hits the same cancellation a different way.")
println(out, "# See the notebook for the full derivation and numerical validation against dblquad.")
println(out)

for k in 0:3
    println("deriving k=$k ...")
    branches = derive_branches(wendland_phi(k, 2))
    for bname in (:generic, :a2_0, :a1_0, :both_0)
        fname = "_wendland_rect_$(bname)_k$(k)"
        body = body_source(branches[bname], [a1, b1, a2, b2])
        println(out, "function $fname(a1, b1, a2, b2)")
        println(out, "    return $body")
        println(out, "end")
        println(out)
    end
    gname = "_wendland_Gclip_k$(k)"
    gbody = body_source(branches.Gclip, [Sym(:y)])
    println(out, "# Plain (non-polar) antiderivative of phi_{$(k),2}, clipped at the kernel's")
    println(out, "# support boundary r=1 -- used for the near-axis Taylor corrections above and the")
    println(out, "# b1-tiny/b2-tiny guard in the dispatcher below. No trig/log, no cancellation risk.")
    println(out, "function $gname(y)")
    println(out, "    return $gbody")
    println(out, "end")
    println(out)
end

println(out, """
# How close a1/a2 (and b1/b2, for the whole-rectangle guard below) can get to 0 relative
# to the opposite edge before the near-axis antiderivative evaluation shared by every
# branch suffers catastrophic cancellation -- k-dependent, and NOT a small effect: a
# dense sweep (4000 log-spaced ratio samples per k, per Phi_i; Phi1/Phi2/Phi3/Phi4 all
# gave identical worst ratios per k, as expected by the a<->b / sin<->cos symmetry of the
# derivation) found the largest *broken* ratio at k=0: ~1e-8, k=1: ~0.0029, k=2: ~0.0145,
# k=3: ~0.0634 -- i.e. the danger zone widens sharply with k (higher-degree polynomial,
# more terms in the cancelling denominator). A single tol=0.01 (safe for k<=1) silently
# produced Inf/NaN for k=2 in real use (every grid resolution of a convergence study,
# not just a contrived case) before this was caught -- see the derivation notebook for
# the full sweep and the incident that found it.
#
# The margin is deliberately NOT generous (~2-2.4x over the swept worst case, not 5x):
# tol does double duty -- it's also the boundary of the Taylor-corrected approximation
# (O(tol^2) error, growing with how far into the zone a given a1/a2 actually is), so a
# wider-than-needed tol doesn't just "waste" some safety margin, it actively degrades
# accuracy for ordinary, non-dangerous inputs that get needlessly routed through the
# approximation instead of the (exact, ~1e-9) generic formula. Confirmed concretely: an
# earlier 5x-margin tol=0.35 for k=3 caused a plain "inside" case, (0.1,0.5,0.2,0.6),
# nowhere near any axis, to pick up a 3.3e-4 error it shouldn't have had at all -- not
# itself unsafe, just needlessly imprecise. k=0/k=1 are left at their original,
# already-validated 0.01 (>=3.4x margin, zero observed failures) rather than narrowed
# further, since there's no accuracy problem there to fix.
const _WENDLAND_NEAR_AXIS_TOL = (0.01, 0.01, 0.03, 0.15)  # indexed by k+1 (k=0:3)

# Dispatches on k (and d=2, the only supported ambient/kernel dimension combination for
# this fast path); each branch calls a separately-compiled, individually type-stable
# function, so despite k/d being runtime fields (not type parameters) on WendlandKernel,
# this dispatcher itself stays type-stable (every branch returns Float64).
function _wendland_rect_first_quadrant(k::Int, d::Int, a1v, b1v, a2v, b2v)
    d == 2 && k in 0:3 || return nothing
    gfun = k == 0 ? _wendland_Gclip_k0 : k == 1 ? _wendland_Gclip_k1 :
           k == 2 ? _wendland_Gclip_k2 : _wendland_Gclip_k3
    tol = _WENDLAND_NEAR_AXIS_TOL[k + 1]
    # Whole-rectangle-near-an-axis guard: if b1 << b2 (so 0 <= a1 < b1 forces the entire
    # x-range close to 0 too, not just a1), phi(sqrt(x^2+y^2)) ~ phi(y) + O(x^2) throughout
    # the strip (the O(x) term vanishes by symmetry), so this is exact to O(b1^3), not just
    # a fallback -- confirmed numerically to agree with the true integral to machine
    # precision for a representative near-degenerate case. No singular angle evaluated at
    # all, so no cancellation risk regardless of how small b1 is. Symmetric for b2 << b1.
    if b1v < tol * b2v
        return max((b1v - a1v) * (gfun(b2v) - gfun(a2v)), 0.0)
    elseif b2v < tol * b1v
        return max((b2v - a2v) * (gfun(b1v) - gfun(a1v)), 0.0)
    end
    a1_tiny = a1v < tol * b2v
    a2_tiny = a2v < tol * b1v
    raw = if k == 0
        a1_tiny && a2_tiny ? _wendland_rect_both_0_k0(a1v, b1v, a2v, b2v) :
        a1_tiny            ? _wendland_rect_a1_0_k0(a1v, b1v, a2v, b2v) :
        a2_tiny            ? _wendland_rect_a2_0_k0(a1v, b1v, a2v, b2v) :
                              _wendland_rect_generic_k0(a1v, b1v, a2v, b2v)
    elseif k == 1
        a1_tiny && a2_tiny ? _wendland_rect_both_0_k1(a1v, b1v, a2v, b2v) :
        a1_tiny            ? _wendland_rect_a1_0_k1(a1v, b1v, a2v, b2v) :
        a2_tiny            ? _wendland_rect_a2_0_k1(a1v, b1v, a2v, b2v) :
                              _wendland_rect_generic_k1(a1v, b1v, a2v, b2v)
    elseif k == 2
        a1_tiny && a2_tiny ? _wendland_rect_both_0_k2(a1v, b1v, a2v, b2v) :
        a1_tiny            ? _wendland_rect_a1_0_k2(a1v, b1v, a2v, b2v) :
        a2_tiny            ? _wendland_rect_a2_0_k2(a1v, b1v, a2v, b2v) :
                              _wendland_rect_generic_k2(a1v, b1v, a2v, b2v)
    elseif k == 3
        a1_tiny && a2_tiny ? _wendland_rect_both_0_k3(a1v, b1v, a2v, b2v) :
        a1_tiny            ? _wendland_rect_a1_0_k3(a1v, b1v, a2v, b2v) :
        a2_tiny            ? _wendland_rect_a2_0_k3(a1v, b1v, a2v, b2v) :
                              _wendland_rect_generic_k3(a1v, b1v, a2v, b2v)
    else
        return nothing
    end
    # phi >= 0 on its support, so the integral is provably nonnegative; clamp away
    # occasional tiny negative floating-point artifacts right at r=1.
    return max(raw, 0.0)
end

# Reduces an arbitrary rectangle (any quadrant, any straddling of the axes) to up to four
# first-quadrant sub-rectangles via radial symmetry: phi(||x||) depends only on |x_1| and
# |x_2|. Returns `nothing` for unsupported (k,d) so the caller can fall back.
function _wendland_rectangle_integral(k::Int, d::Int, a1v, b1v, a2v, b2v)
    a1v < b1v && a2v < b2v ||
        throw(ArgumentError("require a1 < b1 and a2 < b2, got a1=\$a1v, b1=\$b1v, a2=\$a2v, b2=\$b2v"))
    Px = (max(a1v, 0.0), max(b1v, 0.0)); Nx = (max(-b1v, 0.0), max(-a1v, 0.0))
    Py = (max(a2v, 0.0), max(b2v, 0.0)); Ny = (max(-b2v, 0.0), max(-a2v, 0.0))
    total = 0.0
    for (x_lo, x_hi) in (Px, Nx)
        x_hi <= x_lo && continue
        for (y_lo, y_hi) in (Py, Ny)
            y_hi <= y_lo && continue
            piece = _wendland_rect_first_quadrant(k, d, x_lo, x_hi, y_lo, y_hi)
            piece === nothing && return nothing
            total += piece
        end
    end
    return total
end
""")

OUTFILE = joinpath(@__DIR__, "..", "ext", "wendland_rectangle_integral.jl")
open(OUTFILE, "w") do io
    write(io, String(take!(out)))
end
println("wrote $OUTFILE")
