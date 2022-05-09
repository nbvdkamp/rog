in vec4 position;
in vec3 normal;
in vec2 uv;

out vec3 v_normal;
out vec3 v_light_direction;
out vec4 v_base_color;
out vec2 v_uv;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec4 u_base_color;
uniform vec3 u_light_position;

void main() {
    vec4 pos = u_projection * u_view * position;

    v_normal = normal;
    v_light_direction = normalize(u_light_position - vec3(position));
    v_base_color = u_base_color;
    v_uv = uv;

    gl_Position = pos;
}