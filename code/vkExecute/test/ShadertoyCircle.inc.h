// Courtesy https://www.shadertoy.com/view/lllBR7
// No license specified; Defaults to CC BY-NC-SA v3.0 Unported as per Shadertoy
// website terms

static const char *vertexShaderSrc = R"shdrSrc(#version 310 es
precision highp float;
precision highp int;
precision mediump sampler3D;
layout(location = 0) in vec3 inPosition;
void main() {gl_Position = vec4(inPosition, 1.0);}
)shdrSrc";

static const char *fragmentShaderSrc = R"shdrSrc(#version 310 es
precision highp float;
precision highp int;
precision mediump sampler3D;

layout(location = 0) out vec4 outColor;

layout (binding=0) uniform PrimaryUBO {
  uniform vec3 iResolution;
  uniform float iTime;
  uniform vec4 iChannelTime;
  uniform vec4 iMouse;
  uniform vec4 iDate;
  uniform vec3 iChannelResolution[4];
  uniform float iSampleRate;
  uniform int iFrame;
  uniform float iTimeDelta;
  uniform float iFrameRate;
};

void mainImage(out vec4 c, in vec2 f);
void main() {mainImage(outColor, gl_FragCoord.xy);}

float Circle( vec2 uv, vec2 p, float r, float blur )
{
	float d = length(uv - p);
  float c = smoothstep(r, r-blur, d);
  return c;
}

float Hash( float h )
{
  return h = fract(cos(h) * 5422.2465);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
  float c = 0.0;
    
  uv -= .5;
  uv.x *= iResolution.x / iResolution.y;
  float sizer = 1.0;
  float steper = .1;
  for(float i = -sizer; i<sizer; i+=steper)
    for(float j = -sizer; j<sizer; j+=steper)
    {	
      float timer = .5;
      float resetTimer = 7.0;
      if(c<=1.0){
        c += Circle(uv, vec2(i, j),sin(Hash(i))*cos(Hash(j))*(mod(iTime*timer, resetTimer)), sin(Hash(j)));
      }
      else if(c>=1.0)
      {
        c -= Circle(uv, vec2(i, j),sin(Hash(i))*cos(Hash(j))*(mod(iTime*timer, resetTimer)), sin(Hash(j)));     
      }
    }
  fragColor = vec4(vec3(c),1.0);
}
)shdrSrc";

// This is for testing u32 trace counter overflow.
static const char *fragmentShaderTestLargeU64Src = R"shdrSrc(#version 310 es
precision highp float;
precision highp int;
precision mediump sampler3D;

layout(location = 0) out vec4 outColor;

layout (binding=0) uniform PrimaryUBO {
  uniform vec3 iResolution;
  uniform float iTime;
  uniform vec4 iChannelTime;
  uniform vec4 iMouse;
  uniform vec4 iDate;
  uniform vec3 iChannelResolution[4];
  uniform float iSampleRate;
  uniform int iFrame;
  uniform float iTimeDelta;
  uniform float iFrameRate;
};

void mainImage(out vec4 c, in vec2 f);
void main() {mainImage(outColor, gl_FragCoord.xy);}

float Circle( vec2 uv, vec2 p, float r, float blur )
{
	float d = length(uv - p);
  float c = smoothstep(r, r-blur, d);
  return c;
}

float Hash( float h )
{
  return h = fract(cos(h) * 5422.2465);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
  float c = 0.0;
    
  uv -= .5;
  uv.x *= iResolution.x / iResolution.y;
  float sizer = 1.0;
  float steper = .1;
  for (int m = 0; m < 15; m++) {
    for(float i = -sizer; i<sizer; i+=steper)
      for(float j = -sizer; j<sizer; j+=steper)
      {	
        float timer = .5;
        float resetTimer = 7.0;
        if(c<=1.0){
          c += Circle(uv, vec2(i, j),sin(Hash(i))*cos(Hash(j))*(mod(iTime*timer, resetTimer)), sin(Hash(j)));
        }
        else if(c>=1.0)
        {
          c -= Circle(uv, vec2(i, j),sin(Hash(i))*cos(Hash(j))*(mod(iTime*timer, resetTimer)), sin(Hash(j)));     
        }
      }
  }

  fragColor = vec4(vec3(c),1.0);
}
)shdrSrc";