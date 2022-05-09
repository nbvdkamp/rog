out vec4 frag_color;

in vec3 v_normal;
in vec4 v_base_color;
in vec2 v_uv;

uniform sampler2D u_base_color_texture;
uniform bool u_use_texture;

void main() {
    const vec3 light_dir = vec3(0., 1., -.5);
    float light = 0.2 + 0.8 * max(0, dot(v_normal, -normalize(light_dir)));
    vec4 tex_value = u_use_texture ? texture(u_base_color_texture, v_uv) : vec4(1.0);
    frag_color = v_base_color * tex_value * light;
}