in vec4 position;
in vec3 normal;

out vec3 v_normal;

uniform mat4 u_projection;
uniform mat4 u_view;

void main() {
    v_normal = normal;
    gl_Position =  u_projection * u_view * position;
}