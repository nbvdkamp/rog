in vec3 position;
in vec3 normal;
in vec2 uv;

out vec3 v_normal;
out vec3 v_light_direction;
out vec3 v_fragment_position;
out vec2 v_uv;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;
uniform mat3 u_normal_transform;
uniform mat3 u_uv_transform;

void main() {
    v_normal = normalize(u_normal_transform * normal);
    vec4 fragment_pos = u_model * vec4(position, 1.0);
    v_fragment_position = fragment_pos.xyz / fragment_pos.w;
    vec3 uv_homogeneous = u_uv_transform * vec3(uv, 1.0);
    v_uv = uv_homogeneous.xy / uv_homogeneous.z;

    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
}