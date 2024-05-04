out vec4 frag_color;

in vec3 v_normal;
in vec3 v_fragment_position;
in vec2 v_uv;

uniform sampler2D u_base_color_texture;
uniform bool u_use_texture;
uniform vec3 u_light_position;
uniform vec4 u_base_color;

void main() {
    vec3 light_direction = normalize(u_light_position - v_fragment_position);
    float ambient = 0.3;
    float light = ambient + (1.0 - ambient) * max(0.0, dot(normalize(v_normal), light_direction));

    vec4 tex_value = u_use_texture ? texture(u_base_color_texture, v_uv) : vec4(1.0);
    vec4 color = u_base_color * tex_value;

    frag_color = vec4(color.rgb * light, color.a);
}