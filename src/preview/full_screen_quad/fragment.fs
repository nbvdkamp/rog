out vec4 frag_color;

in vec2 v_uv;

uniform sampler2D u_texture;

void main() {
    frag_color = texture(u_texture, v_uv);
}