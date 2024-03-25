use cgmath::{vec2, InnerSpace, Point2, Point3, Vector3};

use super::axis::Axis;

pub fn orthogonal_vector(v: Vector3<f32>) -> Vector3<f32> {
    if v.x == 0.0 {
        Vector3::new(0.0, -v.z, v.y)
    } else if v.y == 0.0 {
        Vector3::new(-v.z, 0.0, v.x)
    } else {
        Vector3::new(-v.y, v.x, 0.0)
    }
}

pub fn spherical_to_cartesian(theta: f32, phi: f32) -> Vector3<f32> {
    let sin_theta = theta.sin();

    return Vector3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, theta.cos() * sin_theta);
}

/// normal must be a unit vector
pub fn reflect(v: Vector3<f32>, normal: Vector3<f32>) -> Vector3<f32> {
    2.0 * normal.dot(v) * normal - v
}

/// normal must be a unit vector
pub fn refract(v: Vector3<f32>, normal: Vector3<f32>, relative_ior: f32, cos_theta_t: f32) -> Vector3<f32> {
    (relative_ior * normal.dot(v) - cos_theta_t) * normal - relative_ior * v
}

pub fn line_axis_plane_intersect(a: Point3<f32>, b: Point3<f32>, axis: Axis, position: f32) -> (f32, Point3<f32>) {
    let i = axis.index();

    let t = (position - a[i]) / (b[i] - a[i]);
    (t, a + t * (b - a))
}

pub fn triangle_area(v0: Point3<f32>, v1: Point3<f32>, v2: Point3<f32>) -> f32 {
    let edge01 = v1 - v0;
    let edge02 = v2 - v0;
    edge01.cross(edge02).magnitude() / 2.0
}

pub fn interpolate_point_on_triangle(
    point: Point2<f32>,
    v0: Point2<f32>,
    v1: Point2<f32>,
    v2: Point2<f32>,
) -> Point2<f32> {
    let edge01 = v1 - v0;
    let edge02 = v2 - v0;
    v0 + point.x * edge01 + point.y * edge02
}

/*
 * The following function is adapted from Blender cycles with this license:
 *
 * Adapted from Open Shading Language with this license:
 *
 * Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
 * All Rights Reserved.
 *
 * Modifications Copyright 2011, Blender Foundation.
 *
 * Port to rust Copyright 2022, Nathan van der Kamp
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of Sony Pictures Imageworks nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#[allow(non_snake_case)]
pub fn ensure_valid_reflection(Ng: Vector3<f32>, I: Vector3<f32>, N: Vector3<f32>) -> Vector3<f32> {
    let R = reflect(I, N);

    /* Reflection rays may always be at least as shallow as the incoming ray. */
    let threshold = 0.01_f32.min(0.9 * Ng.dot(I));

    if Ng.dot(R) >= threshold {
        return N;
    }

    /* Form coordinate system with Ng as the Z axis and N inside the X-Z-plane.
     * The X axis is found by normalizing the component of N that's orthogonal to Ng.
     * The Y axis isn't actually needed.
     */
    let NdotNg = N.dot(Ng);
    let X = (N - NdotNg * Ng).normalize();

    /* Calculate N.z and N.x in the local coordinate system.
     *
     * The goal of this computation is to find a N' that is rotated towards Ng just enough
     * to lift R' above the threshold (here called t), therefore dot(R', Ng) = t.
     *
     * According to the standard reflection equation,
     * this means that we want dot(2*dot(N', I)*N' - I, Ng) = t.
     *
     * Since the Z axis of our local coordinate system is Ng, dot(x, Ng) is just x.z, so we get
     * 2*dot(N', I)*N'.z - I.z = t.
     *
     * The rotation is simple to express in the coordinate system we formed -
     * since N lies in the X-Z-plane, we know that N' will also lie in the X-Z-plane,
     * so N'.y = 0 and therefore dot(N', I) = N'.x*I.x + N'.z*I.z .
     *
     * Furthermore, we want N' to be normalized, so N'.x = sqrt(1 - N'.z^2).
     *
     * With these simplifications,
     * we get the final equation 2*(sqrt(1 - N'.z^2)*I.x + N'.z*I.z)*N'.z - I.z = t.
     *
     * The only unknown here is N'.z, so we can solve for that.
     *
     * The equation has four solutions in general:
     *
     * N'.z = +-sqrt(0.5*(+-sqrt(I.x^2*(I.x^2 + I.z^2 - t^2)) + t*I.z + I.x^2 + I.z^2)/(I.x^2 + I.z^2))
     * We can simplify this expression a bit by grouping terms:
     *
     * a = I.x^2 + I.z^2
     * b = sqrt(I.x^2 * (a - t^2))
     * c = I.z*t + a
     * N'.z = +-sqrt(0.5*(+-b + c)/a)
     *
     * Two solutions can immediately be discarded because they're negative so N' would lie in the
     * lower hemisphere.
     */

    let Ix = I.dot(X);
    let Iz = I.dot(Ng);
    let Ix2 = Ix * Ix;
    let Iz2 = Iz * Iz;
    let a = Ix2 + Iz2;

    let b = (Ix2 * (a - threshold * threshold)).sqrt().max(0.0);
    let c = Iz * threshold + a;

    /* Evaluate both solutions.
     * In many cases one can be immediately discarded (if N'.z would be imaginary or larger than
     * one), so check for that first. If no option is viable (might happen in extreme cases like N
     * being in the wrong hemisphere), give up and return Ng. */
    let fac = 0.5 / a;
    let N1_z2 = fac * (b + c);
    let N2_z2 = fac * (-b + c);
    let mut valid1 = (N1_z2 > 1e-5) && (N1_z2 <= (1.0 + 1e-5));
    let mut valid2 = (N2_z2 > 1e-5) && (N2_z2 <= (1.0 + 1e-5));

    let N_new;
    if valid1 && valid2 {
        /* If both are possible, do the expensive reflection-based check. */
        let N1 = vec2((1.0 - N1_z2).sqrt().max(0.0), N1_z2.sqrt().max(0.0));
        let N2 = vec2((1.0 - N2_z2).sqrt().max(0.0), N2_z2.sqrt().max(0.0));

        let R1 = 2.0 * (N1.x * Ix + N1.y * Iz) * N1.y - Iz;
        let R2 = 2.0 * (N2.x * Ix + N2.y * Iz) * N2.y - Iz;

        valid1 = R1 >= 1e-5;
        valid2 = R2 >= 1e-5;
        if valid1 && valid2 {
            /* If both solutions are valid, return the one with the shallower reflection since it will be
             * closer to the input (if the original reflection wasn't shallow, we would not be in this
             * part of the function). */
            N_new = if R1 < R2 { N1 } else { N2 };
        } else {
            /* If only one reflection is valid (= positive), pick that one. */
            N_new = if R1 > R2 { N1 } else { N2 };
        }
    } else if valid1 || valid2 {
        /* Only one solution passes the N'.z criterium, so pick that one. */
        let Nz2 = if valid1 { N1_z2 } else { N2_z2 };
        N_new = vec2((1.0 - Nz2).sqrt().max(0.0), Nz2.sqrt().max(0.0));
    } else {
        return Ng;
    }

    N_new.x * X + N_new.y * Ng
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{assert_abs_diff_eq, point3, vec3};

    #[test]
    fn basic_reflect() {
        let v = vec3(1.0, 1.0, 0.0);
        let n = vec3(0.0, 1.0, 0.0);
        assert_eq!(reflect(v, n), vec3(-1.0, 1.0, 0.0));
    }

    #[test]
    fn diag_reflect() {
        let v = vec3(1.0, 0.0, 0.0);
        let n = vec3(1.0, 1.0, 0.0).normalize();
        assert_abs_diff_eq!(reflect(v, n), vec3(0.0, 1.0, 0.0), epsilon = 0.0001);
    }

    #[test]
    fn simple_orthogonal_vector() {
        let v = vec3(1.0, 2.0, 3.0);
        let t = orthogonal_vector(v);
        assert_abs_diff_eq!(v.dot(t), 0.0);
    }

    #[test]
    fn simple_line_plane_intersect() {
        let a = point3(-1.0, 0.0, 0.0);
        let b = point3(1.0, 0.0, 0.0);
        let (t, p) = line_axis_plane_intersect(a, b, Axis::X, 0.0);
        assert_abs_diff_eq!(t, 0.5);
        assert_abs_diff_eq!(p, point3(0.0, 0.0, 0.0));
    }

    #[test]
    fn diagonal_line_plane_intersect() {
        let a = point3(-1.0, -1.0, 0.0);
        let b = point3(1.0, 1.0, 0.0);
        let (t, p) = line_axis_plane_intersect(a, b, Axis::X, 0.0);
        assert_abs_diff_eq!(t, 0.5);
        assert_abs_diff_eq!(p, point3(0.0, 0.0, 0.0));
    }
}
