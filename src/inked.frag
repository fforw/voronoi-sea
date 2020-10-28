#version 300 es
precision lowp float;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec4 u_mouse;
uniform vec3 u_palette[8];
uniform float u_shiny[8];

const float pi = 3.141592653589793;
const float tau = pi * 2.0;
const float hpi = pi * 0.5;
const float phi = (1.0+sqrt(5.0))/2.0;

out vec4 outColor;


#define MAX_STEPS 100
#define MAX_DIST 300.
#define SURF_DIST .001

#define ROT(a) mat2(cos(a), -sin(a), sin(a), cos(a))
#define SHEARX(a) mat2(1, 0, sin(a), 1)

////////////////////// NOISE

//	Simplex 3D Noise
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //  x0 = x0 - 0. + 0.0 * C
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1. + 3.0 * C.xxx;

    // Permutations
    i = mod(i, 289.0 );
    vec4 p = permute( permute( permute(
    i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
    dot(p2,x2), dot(p3,x3) ) );
}

float rand(float n){return fract(sin(n) * 43758.5453123);}

// Camera helper

vec3 Camera(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l-p),
    r = normalize(
    cross(
    vec3(0, 1, 0),
    f
    )
    ),
    u = cross(f, r),
    c = p + f * z,
    i = c + uv.x*r + uv.y*u,
    d = normalize(i-p);
    return d;
}


// 2d rotation matrix helper
mat2 Rot(float a) {
    float x = cos(a);
    float y = sin(a);
    return mat2(x, -y, y, x);
}

// RAY MARCHING PRIMITIVES

float smin(float a, float b, float k) {
    float h = clamp(0.5+0.5*(b-a)/k, 0., 1.);
    return mix(b, a, h) - k*h*(1.0-h);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 ab = b-a;
    vec3 ap = p-a;

    float t = dot(ab, ap) / dot(ab, ab);
    t = clamp(t, 0., 1.);

    vec3 c = a + t*ab;

    return length(p-c)-r;
}

float sdCylinder(vec3 p, vec3 a, vec3 b, float r) {
    vec3 ab = b-a;
    vec3 ap = p-a;

    float t = dot(ab, ap) / dot(ab, ab);
    //t = clamp(t, 0., 1.);

    vec3 c = a + t*ab;

    float x = length(p-c)-r;
    float y = (abs(t-.5)-.5)*length(ab);
    float e = length(max(vec2(x, y), 0.));
    float i = min(max(x, y), 0.);

    return e+i;
}

float sdCappedCylinder( vec3 p, float h, float r )
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdSphere(vec3 p, float s)
{
    return length(p)-s;
}

float sdTorus(vec3 p, vec2 r) {
    float x = length(p.xz)-r.x;
    return length(vec2(x, p.y))-r.y;
}

float sdRoundBox(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}


float sdBeam(vec3 p, vec3 c)
{
    return length(p.xz-c.xy)-c.z;
}

float dBox(vec3 p, vec3 s) {
    p = abs(p)-s;
    return length(max(p, 0.))+min(max(p.x, max(p.y, p.z)), 0.);
}

vec2 opUnion(vec2 curr, float d, float id)
{
    if (d < curr.x)
    {
        curr.x = d;
        curr.y = id;
    }

    return curr;
}

vec2 softMinUnion(vec2 curr, float d, float id)
{
    if (d < curr.x)
    {
        curr.x = smin(curr.x, d, 0.5);
        curr.y = id;
    }

    return curr;
}


float sdBoundingBox(vec3 p, vec3 b, float e)
{
    p = abs(p)-b;
    vec3 q = abs(p+e)-e;
    return min(min(
    length(max(vec3(p.x, q.y, q.z), 0.0))+min(max(p.x, max(q.y, q.z)), 0.0),
    length(max(vec3(q.x, p.y, q.z), 0.0))+min(max(q.x, max(p.y, q.z)), 0.0)),
    length(max(vec3(q.x, q.y, p.z), 0.0))+min(max(q.x, max(q.y, p.z)), 0.0));
}

float sdHexPrism( vec3 p, vec2 h )
{
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
    vec2 d = vec2(
    length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
    p.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float shape(float v, float x)
{
    return x > 0.0 ? -abs(v) : abs(v);
}

const mat2 frontPlaneRot = ROT(0.05235987755982988);
const mat2 backPlaneRot = ROT(-0.05235987755982988);
const mat2 sCutRot = ROT(0.88);
const mat2 rotate90 = ROT(1.5707963267948966);
const mat2 rotate60 = ROT(1.0471975511965976);
const mat2 rotate30 = ROT(0.5235987755982988);
const mat2 fourShear = SHEARX(-0.20943951023931953);


const float sin60 = sin(tau/6.0);

vec2 getDistance(vec3 p) {

    float t = u_time * 0.61;

    // ground plane
    float pd = p.y;

    vec2 result = vec2(1e6, 0);

    vec3 p2 = p;
    p2.x = abs(p2.x) - 1.5 + sin(t) * 0.5;

    vec3 n = normalize(vec3(1,cos(t * 1.1),1));
    p2 -= 2.0 * n * min(0.0, dot(p2,n));

    n = normalize(vec3(cos(t),sin(-t * 1.2),0));
    p2 -= 2.0 * n * min(0.0, dot(p2,n));

    n = normalize(vec3(-1,1,1));
    p2 -= 2.0 * n * min(0.0, dot(p2,n));

    n = normalize(vec3(0,sin(t),-1));
    p2 -= 2.0 * n * min(0.0, dot(p2,n));

    n = normalize(vec3(1,cos(-t * 1.3),1));
    p2 -= 2.0 * n * min(0.0, dot(p2,n));


    float c = 10.0;
    vec3 q = p2;//mod(p2+0.5*c,c)-0.5*c;

    float box = dBox(q - vec3(cos(t*2.1),1,0), vec3(0.95)) - 0.2;
    float box2 = dBox(q - vec3(1,cos(t*2.0),0), vec3(0.95)) - 0.2;
    float box3 = dBox(q - vec3(0,-1,0), vec3(0.95)) - 0.2;

    //result = opUnion(result, pd, 3.0);
    result = opUnion(result, box, 1.0);
    result = opUnion(result, box2, 2.0);
    result = opUnion(result, box3, 3.0);

    return result;
}


vec2 rayMarch(vec3 ro, vec3 rd) {


    float dO = 0.;
    float id = 0.0;

    for (int i=0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd*dO;
        vec2 result = getDistance(p);
        float dS = result.x;
        dO += dS;
        id = result.y;
        if (dO > MAX_DIST || abs(dS) < SURF_DIST * 0.001*(dO*.125 + 1.))
        break;
    }

    return vec2(dO, id);
}

vec3 getNormal(vec3 p) {
    float d = getDistance(p).x;
    vec2 e = vec2(.001, 0);

    vec3 n = d - vec3(
        getDistance(p-e.xyy).x,
        getDistance(p-e.yxy).x,
        getDistance(p-e.yyx).x
    );

    return normalize(n);
}


vec3 getPaletteColor(float id)
{
    int last = u_palette.length() - 1;
    //return id < float(last) ? mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id)) : u_palette[last];
    return mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id));
}



void main(void)
{
    vec2 uv = (gl_FragCoord.xy-.5*u_resolution.xy)/u_resolution.y;
    vec2 m = u_mouse.xy/u_resolution.xy;

    float c = 0.4;

    //vec2 tid = floor(uv / c);

    //uv = (mod(uv+0.5*c,c)-0.5*c)/c;

    vec3 col = vec3(0);
//    vec3 ro = vec3(
//    (cos(u_time * 1.7) + cos(u_time)) * 0.8,
//    (sin(u_time * 1.3) - sin(u_time * 1.9)) * 0.8,
//        -10.0 + sin(u_time) * 2.0
//    );

    vec3 ro = vec3(0,3,-8);
    //    ro.yz *= Rot((-m.y + 0.5)* 7.0);
    ro.xz *= Rot((-m.x + 0.5)* 7.0 );


    vec3 lookAt = vec3(0);

    vec3 rd = Camera(uv, ro, lookAt, 1.3);

    vec2 result = rayMarch(ro, rd);

    float d = result.x;

    if (d < MAX_DIST) {
        vec3 p = ro + rd * d;

        vec3 lightPos = ro + vec3(0,2,0);
        vec3 lightDir = normalize(lightPos - p);
        vec3 norm = getNormal(p);

        vec3 lightColor = vec3(1.0);

        float id = result.y;

        // ambient
        vec3 ambient = lightColor * vec3(0.01,0.005,0);

        // diffuse
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 tone = getPaletteColor(id);
        vec3 diffuse = lightColor * (diff * tone);

        // specular
        vec3 viewDir = normalize(ro);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), u_shiny[int(id)]);
        vec3 specular = lightColor * spec * vec3(0.7843,0.8823,0.9451);

        col = clamp((ambient + diffuse + specular), 0.0, 1.0);

//        col =  dsQ * 0.2 + tone * dif * dsQ * 50.0;
    }

    col = pow(col, vec3(1.0/2.2));

    outColor = vec4(
        col,
        1.0
    );

    //outColor = vec4(1,0,1,1);
}
