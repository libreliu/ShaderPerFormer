# Shadertoy-Helper

Usage:
- Place your API key under `/apiKey.txt`
  - Go to https://www.shadertoy.com/myapps for one
- Download all public api accessible shaders with `getShaders.py`
  - Approximately 20000+ shaders

todo:
- [ ] shader sanity check
- [ ] replay
  - [ ] conformance test with reference implementation

## Special reminder

Shadertoy website will have shader ids that **only differ by name**. This can cause trouble for case-insensitive file systems (e.g. NTFS folders configured as case-insensitive). Currently, use Linux when testing, or configure Windows to be case sensitive in the folder under test (and be aware of any weird behaviors of Windows utility programs).

To overcome this, new path scheme is introduced. In the path `N` = Numeric, `L` = Lowercase and `U` = Uppercase.

## Usage

1. clone to `toyDb` folder
2. **outside** `toyDb` folder, use `python -m toyDb.ShaderDB` and `python -m toyDb.ExperimentDB`
   > Reason: This is considered to be a module for NGPP. The limitation of relative import applies.
   >
   > See [here](https://stackoverflow.com/questions/16981921/relative-imports-in-python-3) for more info.

## Shadertoy basics

There are at most 8 render passes as in JSON (or the website itself):
- (**Required**) Image
- Common
  - This is not a pass actually, but will be automatically included into every other shader pieces
- Sound
- Buffer A/B/C/D
- Cubemap A

And the corresponding JSON hierarchy:

- `renderpass` (Each of the render pass have the following)
  - `inputs`
    - `ctype`: The input type
      - `buffer`
      - `keyboard`
      - `texture`
      - `music`
      - `cubemap`
    - `id`: the resource referenced
      - same id means the same buffer
    - `channel`: determines which uniform to be bound to (e.g. `iChannel0`)
    - `sampler`: texture sampler parameters
      - `filter`
        - `mipmap`
      - `warp`
        - `repeat`
      - `vflip`
        - `true`
      - `srgb`
        - `false`
      - `internal`
        - `byte`
  - `outputs`
    - `id`
    - `channel`: seems to be 0 as always
  - `type`: the type of the output resource
    - `image`
    - `buffer`
    - `common`
    - `sound`
    - `cubemap`

### Uniform variables

```
vec3      iResolution            image/buffer       The viewport resolution (z is pixel aspect ratio, usually 1.0)
float     iTime                  image/sound/buffer Current time in seconds
float     iTimeDelta             image/buffer       Time it takes to render a frame, in seconds
int       iFrame                 image/buffer       Current frame
float     iFrameRate             image/buffer       Number of frames rendered per second
float     iChannelTime[4]        image/buffer       Time for channel (if video or sound), in seconds
vec3      iChannelResolution[4]  image/buffer/sound Input texture resolution for each channel
vec4      iMouse                 image/buffer       xy = current pixel coords (if LMB is down). zw = click pixel
sampler2D iChannel{i}            image/buffer/sound Sampler for input textures i
vec4      iDate                  image/buffer/sound Year, month, day, time in seconds in .xyzw
float     iSampleRate            image/buffer/sound The sound sample rate (typically 44100)
```

### Routines

- Image, Buffer A/B/C/D:
  - `void mainImage( out vec4 fragColor, in vec2 fragCoord )`
- Sound
  - `vec2 mainSound( int samp, float time )`
  - the `mainSound()` function returns a vec2 containing the left and right (stereo) sound channel wave data
- Cubemap
  - `void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )`

Example shader amended by Shadertoy (courtesy: [lllBR7 @ Shadertoy](https://shadertoy.com/view/lllBR7)):

```glsl
#version 300 es
#ifdef GL_ES precision highp float;
precision highp int;
precision mediump sampler3D;
#endif
#define HW_PERFORMANCE 1
uniform vec3 iResolution;
uniform float iTime;
uniform float iChannelTime[4];
uniform vec4 iMouse;
uniform vec4 iDate;
uniform float iSampleRate;
uniform vec3 iChannelResolution[4];
uniform int iFrame;
uniform float iTimeDelta;
uniform float iFrameRate;
uniform sampler2D iChannel0;
uniform struct { sampler2D sampler; vec3 size; float time; int loaded; }iCh0;
uniform sampler2D iChannel1;
uniform struct { sampler2D sampler; vec3 size; float time; int loaded; }iCh1;
uniform sampler2D iChannel2;
uniform struct { sampler2D sampler; vec3 size; float time; int loaded; }iCh2;
uniform sampler2D iChannel3;
uniform struct { sampler2D sampler; vec3 size; float time; int loaded; }iCh3;
void mainImage( out vec4 c, in vec2 f );
void st_assert( bool cond );
void st_assert( bool cond, int v );
out vec4 shadertoy_out_color;
void st_assert( bool cond, int v ) {if(!cond){if(v==0)shadertoy_out_color.x=-1.0;else if(v==1)shadertoy_out_color.y=-1.0;else if(v==2)shadertoy_out_color.z=-1.0;else shadertoy_out_color.w=-1.0;}}
void st_assert( bool cond ) {if(!cond)shadertoy_out_color.x=-1.0;}
void main( void ){
  shadertoy_out_color = vec4(1.0,1.0,1.0,1.0);
  vec4 color = vec4(1e20);
  mainImage( color, gl_FragCoord.xy );
  if(shadertoy_out_color.x<0.0)
    color=vec4(1.0,0.0,0.0,1.0);
  if(shadertoy_out_color.y<0.0)
    color=vec4(0.0,1.0,0.0,1.0);
  if(shadertoy_out_color.z<0.0)
    color=vec4(0.0,0.0,1.0,1.0);
  if(shadertoy_out_color.w<0.0)
    color=vec4(1.0,1.0,0.0,1.0);
  shadertoy_out_color = vec4(color.xyz,1.0);
}

float Circle( vec2 uv, vec2 p, float r, float blur ) { float d = length(uv - p); float c = smoothstep(r, r-blur, d); return c; }
float Hash( float h ) { return h = fract(cos(h) * 5422.2465); }
void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
  vec2 uv = fragCoord.xy / iResolution.xy;
  float c = 0.0;
  uv -= .5;
  uv.x *= iResolution.x / iResolution.y;
  float sizer = 1.0;
  float steper = .1;
  for(float i = -sizer; i=1.0) {
    c -= Circle(uv, vec2(i, j),sin(Hash(i))*cos(Hash(j))*(mod(iTime*timer, resetTimer)), sin(Hash(j)));
  }
}

} fragColor = vec4( vec3(c),1.0); } #define Kalgbo
```

```
#version 300 es #ifdef GL_ES precision highp float; precision highp int; precision mediump sampler3D; #endif layout(location = 0) in vec2 pos; void main() { gl_Position = vec4(pos.xy,0.0,1.0); } #define Kjme9kd
```