var Demo=function(n){var e={};function t(r){if(e[r])return e[r].exports;var o=e[r]={i:r,l:!1,exports:{}};return n[r].call(o.exports,o,o.exports,t),o.l=!0,o.exports}return t.m=n,t.c=e,t.d=function(n,e,r){t.o(n,e)||Object.defineProperty(n,e,{enumerable:!0,get:r})},t.r=function(n){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(n,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(n,"__esModule",{value:!0})},t.t=function(n,e){if(1&e&&(n=t(n)),8&e)return n;if(4&e&&"object"==typeof n&&n&&n.__esModule)return n;var r=Object.create(null);if(t.r(r),Object.defineProperty(r,"default",{enumerable:!0,value:n}),2&e&&"string"!=typeof n)for(var o in n)t.d(r,o,function(e){return n[e]}.bind(null,o));return r},t.n=function(n){var e=n&&n.__esModule?function(){return n.default}:function(){return n};return t.d(e,"a",e),e},t.o=function(n,e){return Object.prototype.hasOwnProperty.call(n,e)},t.p="",t(t.s=4)}([function(n,e,t){(function(e){(function(){var t,r,o,a,i,c;"undefined"!=typeof performance&&null!==performance&&performance.now?n.exports=function(){return performance.now()}:null!=e&&e.hrtime?(n.exports=function(){return(t()-i)/1e6},r=e.hrtime,a=(t=function(){var n;return 1e9*(n=r())[0]+n[1]})(),c=1e9*e.uptime(),i=a-c):Date.now?(n.exports=function(){return Date.now()-o},o=Date.now()):(n.exports=function(){return(new Date).getTime()-o},o=(new Date).getTime())}).call(this)}).call(this,t(2))},function(n,e,t){},function(n,e){var t,r,o=n.exports={};function a(){throw new Error("setTimeout has not been defined")}function i(){throw new Error("clearTimeout has not been defined")}function c(n){if(t===setTimeout)return setTimeout(n,0);if((t===a||!t)&&setTimeout)return t=setTimeout,setTimeout(n,0);try{return t(n,0)}catch(e){try{return t.call(null,n,0)}catch(e){return t.call(this,n,0)}}}!function(){try{t="function"==typeof setTimeout?setTimeout:a}catch(n){t=a}try{r="function"==typeof clearTimeout?clearTimeout:i}catch(n){r=i}}();var l,u=[],f=!1,s=-1;function v(){f&&l&&(f=!1,l.length?u=l.concat(u):s=-1,u.length&&p())}function p(){if(!f){var n=c(v);f=!0;for(var e=u.length;e;){for(l=u,u=[];++s<e;)l&&l[s].run();s=-1,e=u.length}l=null,f=!1,function(n){if(r===clearTimeout)return clearTimeout(n);if((r===i||!r)&&clearTimeout)return r=clearTimeout,clearTimeout(n);try{r(n)}catch(e){try{return r.call(null,n)}catch(e){return r.call(this,n)}}}(n)}}function d(n,e){this.fun=n,this.array=e}function m(){}o.nextTick=function(n){var e=new Array(arguments.length-1);if(arguments.length>1)for(var t=1;t<arguments.length;t++)e[t-1]=arguments[t];u.push(new d(n,e)),1!==u.length||f||c(p)},d.prototype.run=function(){this.fun.apply(null,this.array)},o.title="browser",o.browser=!0,o.env={},o.argv=[],o.version="",o.versions={},o.on=m,o.addListener=m,o.once=m,o.off=m,o.removeListener=m,o.removeAllListeners=m,o.emit=m,o.prependListener=m,o.prependOnceListener=m,o.listeners=function(n){return[]},o.binding=function(n){throw new Error("process.binding is not supported")},o.cwd=function(){return"/"},o.chdir=function(n){throw new Error("process.chdir is not supported")},o.umask=function(){return 0}},,function(n,e,t){"use strict";t.r(e);t(1);var r=t(0),o=t.n(r);function a(n,e){for(var t=0;t<e.length;t++){var r=e[t];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(n,r.key,r)}}var i=/^(#)?([0-9a-f]+)$/i;function c(n){var e=n.toString(16);return 1===e.length?"0"+e:e}function l(n,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?n+6*(e-n)*t:t<.5?e:t<2/3?n+(e-n)*(2/3-t)*6:n}var u,f,s,v,p,d,m,x,h,y,g,b,w,_=function(){function n(e,t,r){!function(n,e){if(!(n instanceof e))throw new TypeError("Cannot call a class as a function")}(this,n),this.r=void 0,this.g=void 0,this.b=void 0,this.r=e,this.g=t,this.b=r}var e,t,r;return e=n,r=[{key:"validate",value:function(e){var t;if("string"!=typeof e||!(t=i.exec(e)))return null;var r=t[2];return 3===r.length?new n(17*parseInt(r[0],16),17*parseInt(r[1],16),17*parseInt(r[2],16)):6===r.length?new n(parseInt(r.substring(0,2),16),parseInt(r.substring(2,4),16),parseInt(r.substring(4,6),16)):null}},{key:"from",value:function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:1;if(Array.isArray(e)){for(var r=e.length,o=new Float32Array(3*r),a=t/255,i=0,c=0;c<r;c++){var l=n.from(e[c]);o[i++]=l.r*a,o[i++]=l.g*a,o[i++]=l.b*a}return o}var u=n.validate(e);if(!u)throw new Error("Invalid color "+e);return u.r*=t,u.g*=t,u.b*=t,u}},{key:"fromHSL",value:function(e,t,r){var o,a,i;if(t<=0)o=a=i=r;else{var c=r<.5?r*(1+t):r+t-r*t,u=2*r-c;o=l(u,c,e+1/3),a=l(u,c,e),i=l(u,c,e-1/3)}return new n(Math.round(255*o),Math.round(255*a),Math.round(255*i))}}],(t=[{key:"mix",value:function(e,t,r){return r||(r=new n),r.r=this.r+(e.r-this.r)*t|0,r.g=this.g+(e.g-this.g)*t|0,r.b=this.b+(e.b-this.b)*t|0,r}},{key:"multiply",value:function(e,t){return t||(t=new n),t.r=this.r*e,t.g=this.g*e,t.b=this.b*e,t}},{key:"toRGBHex",value:function(){return"#"+c(this.r)+c(this.g)+c(this.b)}},{key:"toHex",value:function(){return(this.r<<16)+(this.g<<8)+this.b}}])&&a(e.prototype,t),r&&a(e,r),n}(),z=(Math.sqrt(5),Math.PI,{width:0,height:0}),S=0,T=0;function A(){var n=-16&window.innerWidth,e=0|window.innerHeight;z.width=n,z.height=e,u.width=n,u.height=e,S=n/2,T=e/2,f.viewport(0,0,u.width,u.height)}function D(n,e,t){var r=n.createShader(e);if(n.shaderSource(r,t),n.compileShader(r),n.getShaderParameter(r,n.COMPILE_STATUS))return r;console.error(n.getShaderInfoLog(r)),n.deleteShader(r)}function C(n){var e=y?1:-1;f.uniform1f(p,o()()/1e3),f.uniform2f(d,z.width,z.height),f.uniform4f(m,S,z.height-T,g*e,(z.height-b)*e),f.clearColor(0,0,0,0),f.clear(f.COLOR_BUFFER_BIT);var t=f.TRIANGLES;f.drawArrays(t,0,6),requestAnimationFrame(C)}function R(n){y&&(S=n.clientX-w.left+self.pageXOffset,T=n.clientY-w.top+self.pageYOffset)}function I(n){y=!0,g=n.clientX-w.left+self.pageXOffset,b=n.clientY-w.top+self.pageYOffset,S=g,T=b}function O(n){y=!1}window.onload=function(){if(u=document.getElementById("screen"),!(f=u.getContext("webgl2")))return u.parentNode.removeChild(u),n="Cannot run shader. Your browser does not support WebGL2.",void(document.getElementById("out").innerHTML="<p>"+n+"</p>");var n,e=D(f,f.VERTEX_SHADER,"#version 300 es\n#define GLSLIFY 1\n\n// an attribute is an input (in) to a vertex shader.\n// It will receive data from a buffer\nin vec4 a_position;\n\n// all shaders have a main function\nvoid main() {\n\n    // gl_Position is a special variable a vertex shader\n    // is responsible for setting\n    gl_Position = a_position;\n}\n"),t=D(f,f.FRAGMENT_SHADER,"#version 300 es\nprecision lowp float;\n#define GLSLIFY 1\n\nuniform float u_time;\nuniform vec2 u_resolution;\nuniform vec4 u_mouse;\nuniform vec3 u_palette[8];\nuniform float u_shiny[8];\n\nconst float pi = 3.141592653589793;\nconst float tau = pi * 2.0;\nconst float hpi = pi * 0.5;\nconst float phi = (1.0+sqrt(5.0))/2.0;\n\nout vec4 outColor;\n\n#define MAX_STEPS 100\n#define MAX_DIST 75.\n#define SURF_DIST .001\n\n#define ROT(a) mat2(cos(a), -sin(a), sin(a), cos(a))\n#define SHEARX(a) mat2(1, 0, sin(a), 1)\n\n////////////////////// NOISE\n\n//\tSimplex 3D Noise\n//\tby Ian McEwan, Ashima Arts\n//\nvec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}\nvec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}\n\nfloat snoise(vec3 v){\n    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;\n    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);\n\n    // First corner\n    vec3 i  = floor(v + dot(v, C.yyy) );\n    vec3 x0 =   v - i + dot(i, C.xxx) ;\n\n    // Other corners\n    vec3 g = step(x0.yzx, x0.xyz);\n    vec3 l = 1.0 - g;\n    vec3 i1 = min( g.xyz, l.zxy );\n    vec3 i2 = max( g.xyz, l.zxy );\n\n    //  x0 = x0 - 0. + 0.0 * C\n    vec3 x1 = x0 - i1 + 1.0 * C.xxx;\n    vec3 x2 = x0 - i2 + 2.0 * C.xxx;\n    vec3 x3 = x0 - 1. + 3.0 * C.xxx;\n\n    // Permutations\n    i = mod(i, 289.0 );\n    vec4 p = permute( permute( permute(\n    i.z + vec4(0.0, i1.z, i2.z, 1.0 ))\n    + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))\n    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));\n\n    // Gradients\n    // ( N*N points uniformly over a square, mapped onto an octahedron.)\n    float n_ = 1.0/7.0; // N=7\n    vec3  ns = n_ * D.wyz - D.xzx;\n\n    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)\n\n    vec4 x_ = floor(j * ns.z);\n    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)\n\n    vec4 x = x_ *ns.x + ns.yyyy;\n    vec4 y = y_ *ns.x + ns.yyyy;\n    vec4 h = 1.0 - abs(x) - abs(y);\n\n    vec4 b0 = vec4( x.xy, y.xy );\n    vec4 b1 = vec4( x.zw, y.zw );\n\n    vec4 s0 = floor(b0)*2.0 + 1.0;\n    vec4 s1 = floor(b1)*2.0 + 1.0;\n    vec4 sh = -step(h, vec4(0.0));\n\n    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;\n    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;\n\n    vec3 p0 = vec3(a0.xy,h.x);\n    vec3 p1 = vec3(a0.zw,h.y);\n    vec3 p2 = vec3(a1.xy,h.z);\n    vec3 p3 = vec3(a1.zw,h.w);\n\n    //Normalise gradients\n    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));\n    p0 *= norm.x;\n    p1 *= norm.y;\n    p2 *= norm.z;\n    p3 *= norm.w;\n\n    // Mix final noise value\n    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);\n    m = m * m;\n    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),\n    dot(p2,x2), dot(p3,x3) ) );\n}\n\nfloat rand(float n){return fract(sin(n) * 43758.5453123);}\n\n// Camera helper\n\nvec3 Camera(vec2 uv, vec3 p, vec3 l, float z) {\n    vec3 f = normalize(l-p),\n    r = normalize(\n    cross(\n    vec3(0, 1, 0),\n    f\n    )\n    ),\n    u = cross(f, r),\n    c = p + f * z,\n    i = c + uv.x*r + uv.y*u,\n    d = normalize(i-p);\n    return d;\n}\n\n// 2d rotation matrix helper\nmat2 Rot(float a) {\n    float x = cos(a);\n    float y = sin(a);\n    return mat2(x, -y, y, x);\n}\n\n// RAY MARCHING PRIMITIVES\n\nfloat smin(float a, float b, float k) {\n    float h = clamp(0.5+0.5*(b-a)/k, 0., 1.);\n    return mix(b, a, h) - k*h*(1.0-h);\n}\n\nfloat sdCapsule(vec3 p, vec3 a, vec3 b, float r) {\n    vec3 ab = b-a;\n    vec3 ap = p-a;\n\n    float t = dot(ab, ap) / dot(ab, ab);\n    t = clamp(t, 0., 1.);\n\n    vec3 c = a + t*ab;\n\n    return length(p-c)-r;\n}\n\nfloat sdCylinder(vec3 p, vec3 a, vec3 b, float r) {\n    vec3 ab = b-a;\n    vec3 ap = p-a;\n\n    float t = dot(ab, ap) / dot(ab, ab);\n    //t = clamp(t, 0., 1.);\n\n    vec3 c = a + t*ab;\n\n    float x = length(p-c)-r;\n    float y = (abs(t-.5)-.5)*length(ab);\n    float e = length(max(vec2(x, y), 0.));\n    float i = min(max(x, y), 0.);\n\n    return e+i;\n}\n\nfloat sdCappedCylinder( vec3 p, float h, float r )\n{\n    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);\n    return min(max(d.x,d.y),0.0) + length(max(d,0.0));\n}\n\nfloat sdSphere(vec3 p, float s)\n{\n    return length(p)-s;\n}\n\nfloat sdTorus(vec3 p, vec2 r) {\n    float x = length(p.xz)-r.x;\n    return length(vec2(x, p.y))-r.y;\n}\n\nfloat sdRoundBox(vec3 p, vec3 b, float r)\n{\n    vec3 q = abs(p) - b;\n    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;\n}\n\nfloat sdBeam(vec3 p, vec3 c)\n{\n    return length(p.xz-c.xy)-c.z;\n}\n\nfloat dBox(vec3 p, vec3 s) {\n    p = abs(p)-s;\n    return length(max(p, 0.))+min(max(p.x, max(p.y, p.z)), 0.);\n}\n\nvec2 opUnion(vec2 curr, float d, float id)\n{\n    if (d < curr.x)\n    {\n        curr.x = d;\n        curr.y = id;\n    }\n\n    return curr;\n}\n\nvec2 softMinUnion(vec2 curr, float d, float id)\n{\n    if (d < curr.x)\n    {\n        curr.x = smin(curr.x, d, 0.5);\n        curr.y = id;\n    }\n\n    return curr;\n}\n\nfloat sdBoundingBox(vec3 p, vec3 b, float e)\n{\n    p = abs(p)-b;\n    vec3 q = abs(p+e)-e;\n    return min(min(\n    length(max(vec3(p.x, q.y, q.z), 0.0))+min(max(p.x, max(q.y, q.z)), 0.0),\n    length(max(vec3(q.x, p.y, q.z), 0.0))+min(max(q.x, max(p.y, q.z)), 0.0)),\n    length(max(vec3(q.x, q.y, p.z), 0.0))+min(max(q.x, max(q.y, p.z)), 0.0));\n}\n\nfloat sdHexPrism( vec3 p, vec2 h )\n{\n    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);\n    p = abs(p);\n    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;\n    vec2 d = vec2(\n    length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),\n    p.z-h.y );\n    return min(max(d.x,d.y),0.0) + length(max(d,0.0));\n}\n\nfloat shape(float v, float x)\n{\n    return x > 0.0 ? -abs(v) : abs(v);\n}\n\nconst mat2 frontPlaneRot = ROT(0.05235987755982988);\nconst mat2 backPlaneRot = ROT(-0.05235987755982988);\nconst mat2 sCutRot = ROT(0.88);\nconst mat2 rotate90 = ROT(1.5707963267948966);\nconst mat2 rotate60 = ROT(1.0471975511965976);\nconst mat2 rotate30 = ROT(0.5235987755982988);\nconst mat2 fourShear = SHEARX(-0.20943951023931953);\n\nconst float sin60 = sin(tau/6.0);\nvec2 N22 (vec2 p) {\n    vec3 a = fract(p.xyx*vec3(123.34, 234.34, 345.65));\n    a += dot(a, a+34.45);\n    return fract(vec2(a.x*a.y, a.y*a.z));\n}\n\nfloat atan2(in float y, in float x)\n{\n    return abs(x) > abs(y) ? hpi - atan(x,y) : atan(y,x);\n}\n\nfloat getHeight(vec2 uv)\n{\n    vec2 gv = fract(uv) - 0.5;\n    vec2 id = floor(uv);\n\n    vec2 cid = vec2(0);\n    float minDist = 1e6;\n\n    float t = u_time;\n\n    for (float y = -1.0; y <= 1.0; y++)\n    {\n        for (float x = -1.0; x <= 1.0; x++)\n        {\n            vec2 off = vec2(x,y);\n            vec2 n = N22(id + off);\n\n            vec2 p = off + sin( n * t) * 0.5;\n            float d = length(gv - p);\n            if (d < minDist)\n            {\n                minDist = d;\n                cid = id + off;\n            }\n        }\n    }\n\n    return minDist * minDist * minDist;\n}\n\nfloat getBaseSea(vec2 p, float t)\n{\n    return sin(p.y * 0.37 - t) * 0.7 - sin(p.x * 0.41 - t) * 0.5;\n}\n\nvec2 getDistance(vec3 p) {\n\n    float t = u_time * 0.61;\n\n    // ground plane\n\n    float pd = p.y -getHeight(p.xz) * 0.4 + getBaseSea(p.xz, t);\n\n    vec2 result = vec2(pd, 3.0);\n\n    float box = dBox(p - vec3(0,-getBaseSea(vec2(0,9), t),9), vec3(1));\n\n    result = opUnion(result, box, 1.0);\n\n    return result;\n}\n\nvec2 rayMarch(vec3 ro, vec3 rd) {\n\n    float dO = 0.;\n    float id = 0.0;\n\n    for (int i=0; i < MAX_STEPS; i++) {\n        vec3 p = ro + rd*dO;\n        vec2 result = getDistance(p);\n        float dS = result.x;\n        dO += dS;\n        id = result.y;\n        if (dO > MAX_DIST || abs(dS) < SURF_DIST * 0.001*(dO*.125 + 1.))\n        break;\n    }\n\n    return vec2(dO, id);\n}\n\nvec3 getNormal(vec3 p) {\n    float d = getDistance(p).x;\n    vec2 e = vec2(.001, 0);\n\n    vec3 n = d - vec3(\n        getDistance(p-e.xyy).x,\n        getDistance(p-e.yxy).x,\n        getDistance(p-e.yyx).x\n    );\n\n    return normalize(n);\n}\n\nvec3 getPaletteColor(float id)\n{\n    int last = u_palette.length() - 1;\n    //return id < float(last) ? mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id)) : u_palette[last];\n    return mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id));\n}\n\nvec3 applyFog(\n    in vec3  rgb,      // original color of the pixel\n    in float distance, // camera to point distance\n    in vec3  rayOri,   // camera position\n    in vec3  rayDir,\n    in vec3 p     // camera to point vector\n)\n{\n    float pos = p.z;\n\n    float c = 0.01  ;\n    float b = 0.95;// + sin((pos + p.x * sin(pos * 0.27)) * 0.31 ) * 0.15 + sin(pos * 0.17 ) * 0.15;\n\n    float fogAmount = c * exp(-rayOri.y*b) * (1.0-exp( -distance*rayDir.y*b ))/rayDir.y;\n    vec3  fogColor  = vec3(1);\n    return mix( rgb, fogColor, fogAmount );\n}\n\nvoid main(void)\n{\n    vec2 uv = (gl_FragCoord.xy-.5*u_resolution.xy)/u_resolution.y;\n    vec2 m = u_mouse.xy/u_resolution.xy;\n\n    vec3 col = vec3(0);\n    vec3 ro = vec3(\n        0,\n        3,\n        -8\n    );\n\n    ro.yz *= Rot((-m.y + 0.5)* 7.0);\n    ro.xz *= Rot((-m.x + 0.5)* 7.0);\n\n    vec3 lookAt = vec3(0);\n\n    vec3 rd = Camera(uv, ro, lookAt, 1.3);\n\n    vec2 result = rayMarch(ro, rd);\n\n    float d = result.x;\n\n    vec3 p = ro + rd * d;\n    if (d < MAX_DIST) {\n\n        vec3 lightPos = ro + vec3(0,1,0);\n        vec3 lightDir = normalize(lightPos - p);\n        vec3 norm = getNormal(p);\n\n        vec3 lightColor = vec3(1);\n\n        float id = result.y;\n\n        // ambient\n        vec3 ambient = lightColor * vec3(0.001);\n\n        // diffuse\n        float diff = max(dot(norm, lightDir), 0.0);\n        vec3 tone = getPaletteColor(id);\n\n        if (id == 4.0)\n        {\n            tone *= snoise(p + vec3(0,0, u_time * 10.0)) * 0.5;\n        }\n\n        vec3 diffuse = lightColor * (diff * tone);\n\n        // specular\n        vec3 viewDir = normalize(ro);\n        vec3 reflectDir = reflect(-lightDir, norm);\n        float spec = pow(max(dot(viewDir, reflectDir), 0.0), u_shiny[int(id)]);\n        vec3 specular = lightColor * spec * vec3(0.7843,0.8823,0.9451) * 0.1;\n\n        col = (ambient + diffuse + specular);\n\n    }\n    col = applyFog(col, d, ro, rd, p);\n\n    col = pow(col, vec3(1.0/2.2));\n\n    outColor = vec4(\n        col,\n        1.0\n    );\n\n    //outColor = vec4(1,0,1,1);\n}\n");v=function(n,e,t){var r=n.createProgram();if(n.attachShader(r,e),n.attachShader(r,t),n.linkProgram(r),n.getProgramParameter(r,n.LINK_STATUS))return r;console.error(n.getProgramInfoLog(r)),n.deleteProgram(r)}(f,e,t);var r=f.getAttribLocation(v,"a_position"),o=f.createBuffer();f.bindBuffer(f.ARRAY_BUFFER,o);f.bufferData(f.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,1,1,-1]),f.STATIC_DRAW),s=f.createVertexArray(),f.bindVertexArray(s),f.enableVertexAttribArray(r);var a=f.FLOAT;f.vertexAttribPointer(r,2,a,!1,0,0),A(),f.viewport(0,0,f.canvas.width,f.canvas.height),p=f.getUniformLocation(v,"u_time"),d=f.getUniformLocation(v,"u_resolution"),m=f.getUniformLocation(v,"u_mouse"),x=f.getUniformLocation(v,"u_palette"),h=f.getUniformLocation(v,"u_shiny"),f.useProgram(v),f.bindVertexArray(s),window.addEventListener("resize",A,!0),u.addEventListener("mousemove",R,!0),u.addEventListener("mousedown",I,!0),document.addEventListener("mouseup",O,!0),w=document.getElementById("screen").getBoundingClientRect();var i=_.from(["#000","#fff","#c02","#00244f","#004d9d","#010101","#4c3a25","#f0f"],1);f.uniform3fv(x,i),f.uniform1fv(h,new Float32Array([2,1e3,2,2,2,2,2,2])),requestAnimationFrame(C)}}]);
//# sourceMappingURL=bundle-main-8864ff4c43e7c7074cbb.js.map