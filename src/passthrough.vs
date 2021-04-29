in vec4 position;

uniform mat4 u_projection;
uniform mat4 u_view;

void main() {
    gl_Position =  u_projection * u_view * position;
}