out vec4 frag_color;

in vec3 v_normal;
in vec4 v_base_color;

void main() {
    const vec3 light_dir = vec3(0., 1., -.5);
    frag_color = v_base_color * dot(v_normal, -normalize(light_dir));
}