in vec2 position;
in vec2 uv;

uniform vec2 u_scale;
uniform vec2 u_translation;

out vec2 v_uv;

void main() {
    v_uv = uv;
    gl_Position = vec4((position * u_scale) + u_translation, 0.0, 1.0);
}