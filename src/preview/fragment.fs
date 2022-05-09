out vec4 frag_color;

in vec3 v_normal;
in vec3 v_light_direction;
in vec4 v_base_color;
in vec2 v_uv;

uniform sampler2D u_base_color_texture;
uniform bool u_use_texture;

void main() {
    float ambient = 0.3;
    float light = ambient + (1.0 - ambient) * max(0.0, dot(normalize(v_normal), normalize(v_light_direction)));

    vec4 tex_value = u_use_texture ? texture(u_base_color_texture, v_uv) : vec4(1.0);

    frag_color = v_base_color * tex_value * light;
}