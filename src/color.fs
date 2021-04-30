out vec4 frag_color;

in vec3 v_normal;

void main() {
    const vec3 color = vec3(.6, .6, .6);
    const vec3 light_dir = vec3(0., -1., -.5);
    frag_color = vec4(color * dot(v_normal, -normalize(light_dir)), 1.);
}