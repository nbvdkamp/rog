in vec4 position;
in vec3 normal;

out vec3 v_normal;
out vec4 v_base_color;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec4 u_base_color;

void main() {
    v_normal = normal;
    v_base_color = u_base_color;
    gl_Position =  u_projection * u_view * position;
}