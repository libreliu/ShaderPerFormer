# lllBR7 Shadertoy; By default CC BY-NC-SA

shaderId = "lllBR7"
fragmentSpv = '''; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 209
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outColor %gl_FragCoord %_
               OpExecutionMode %main OriginUpperLeft
               OpSource ESSL 310
               OpName %main "main"
               OpName %Circle_vf2_vf2_f1_f1_ "Circle(vf2;vf2;f1;f1;"
               OpName %uv "uv"
               OpName %p "p"
               OpName %r "r"
               OpName %blur "blur"
               OpName %Hash_f1_ "Hash(f1;"
               OpName %h "h"
               OpName %mainImage_vf4_vf2_ "mainImage(vf4;vf2;"
               OpName %fragColor "fragColor"
               OpName %fragCoord "fragCoord"
               OpName %outColor "outColor"
               OpName %gl_FragCoord "gl_FragCoord"
               OpName %param "param"
               OpName %param_0 "param"
               OpName %d "d"
               OpName %c "c"
               OpName %uv_0 "uv"
               OpName %PrimaryUBO "PrimaryUBO"
               OpMemberName %PrimaryUBO 0 "iResolution"
               OpMemberName %PrimaryUBO 1 "iTime"
               OpMemberName %PrimaryUBO 2 "iChannelTime"
               OpMemberName %PrimaryUBO 3 "iMouse"
               OpMemberName %PrimaryUBO 4 "iDate"
               OpMemberName %PrimaryUBO 5 "iSampleRate"
               OpMemberName %PrimaryUBO 6 "iChannelResolution"
               OpMemberName %PrimaryUBO 7 "iFrame"
               OpMemberName %PrimaryUBO 8 "iTimeDelta"
               OpMemberName %PrimaryUBO 9 "iFrameRate"
               OpName %_ ""
               OpName %c_0 "c"
               OpName %sizer "sizer"
               OpName %steper "steper"
               OpName %i "i"
               OpName %j "j"
               OpName %timer "timer"
               OpName %resetTimer "resetTimer"
               OpName %param_1 "param"
               OpName %param_2 "param"
               OpName %param_3 "param"
               OpName %param_4 "param"
               OpName %param_5 "param"
               OpName %param_6 "param"
               OpName %param_7 "param"
               OpName %param_8 "param"
               OpName %param_9 "param"
               OpName %param_10 "param"
               OpName %param_11 "param"
               OpName %param_12 "param"
               OpName %param_13 "param"
               OpName %param_14 "param"
               OpDecorate %outColor Location 0
               OpDecorate %gl_FragCoord BuiltIn FragCoord
               OpDecorate %_arr_float_uint_4 ArrayStride 16
               OpDecorate %_arr_v3float_uint_4 ArrayStride 16
               OpMemberDecorate %PrimaryUBO 0 Offset 0
               OpMemberDecorate %PrimaryUBO 1 Offset 12
               OpMemberDecorate %PrimaryUBO 2 Offset 16
               OpMemberDecorate %PrimaryUBO 3 Offset 80
               OpMemberDecorate %PrimaryUBO 4 Offset 96
               OpMemberDecorate %PrimaryUBO 5 Offset 112
               OpMemberDecorate %PrimaryUBO 6 Offset 128
               OpMemberDecorate %PrimaryUBO 7 Offset 192
               OpMemberDecorate %PrimaryUBO 8 Offset 196
               OpMemberDecorate %PrimaryUBO 9 Offset 200
               OpDecorate %PrimaryUBO Block
               OpDecorate %_ DescriptorSet 0
               OpDecorate %_ Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Function_float = OpTypePointer Function %float
         %10 = OpTypeFunction %float %_ptr_Function_v2float %_ptr_Function_v2float %_ptr_Function_float %_ptr_Function_float
         %17 = OpTypeFunction %float %_ptr_Function_float
    %v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
         %23 = OpTypeFunction %void %_ptr_Function_v4float %_ptr_Function_v2float
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %outColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%float_5422_24658 = OpConstant %float 5422.24658
    %v3float = OpTypeVector %float 3
       %uint = OpTypeInt 32 0
     %uint_4 = OpConstant %uint 4
%_arr_float_uint_4 = OpTypeArray %float %uint_4
%_arr_v3float_uint_4 = OpTypeArray %v3float %uint_4
        %int = OpTypeInt 32 1
 %PrimaryUBO = OpTypeStruct %v3float %float %_arr_float_uint_4 %v4float %v4float %float %_arr_v3float_uint_4 %int %float %float
%_ptr_Uniform_PrimaryUBO = OpTypePointer Uniform %PrimaryUBO
          %_ = OpVariable %_ptr_Uniform_PrimaryUBO Uniform
      %int_0 = OpConstant %int 0
%_ptr_Uniform_v3float = OpTypePointer Uniform %v3float
    %float_0 = OpConstant %float 0
  %float_0_5 = OpConstant %float 0.5
     %uint_0 = OpConstant %uint 0
%_ptr_Uniform_float = OpTypePointer Uniform %float
     %uint_1 = OpConstant %uint 1
    %float_1 = OpConstant %float 1
%float_0_100000001 = OpConstant %float 0.100000001
       %bool = OpTypeBool
    %float_7 = OpConstant %float 7
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
      %param = OpVariable %_ptr_Function_v4float Function
    %param_0 = OpVariable %_ptr_Function_v2float Function
         %34 = OpLoad %v4float %gl_FragCoord
         %35 = OpVectorShuffle %v2float %34 %34 0 1
               OpStore %param_0 %35
         %36 = OpFunctionCall %void %mainImage_vf4_vf2_ %param %param_0
         %37 = OpLoad %v4float %param
               OpStore %outColor %37
               OpReturn
               OpFunctionEnd
%Circle_vf2_vf2_f1_f1_ = OpFunction %float None %10
         %uv = OpFunctionParameter %_ptr_Function_v2float
          %p = OpFunctionParameter %_ptr_Function_v2float
          %r = OpFunctionParameter %_ptr_Function_float
       %blur = OpFunctionParameter %_ptr_Function_float
         %16 = OpLabel
          %d = OpVariable %_ptr_Function_float Function
          %c = OpVariable %_ptr_Function_float Function
         %39 = OpLoad %v2float %uv
         %40 = OpLoad %v2float %p
         %41 = OpFSub %v2float %39 %40
         %42 = OpExtInst %float %1 Length %41
               OpStore %d %42
         %44 = OpLoad %float %r
         %45 = OpLoad %float %r
         %46 = OpLoad %float %blur
         %47 = OpFSub %float %45 %46
         %48 = OpLoad %float %d
         %49 = OpExtInst %float %1 SmoothStep %44 %47 %48
               OpStore %c %49
         %50 = OpLoad %float %c
               OpReturnValue %50
               OpFunctionEnd
   %Hash_f1_ = OpFunction %float None %17
          %h = OpFunctionParameter %_ptr_Function_float
         %20 = OpLabel
         %53 = OpLoad %float %h
         %54 = OpExtInst %float %1 Cos %53
         %56 = OpFMul %float %54 %float_5422_24658
         %57 = OpExtInst %float %1 Fract %56
               OpStore %h %57
               OpReturnValue %57
               OpFunctionEnd
%mainImage_vf4_vf2_ = OpFunction %void None %23
  %fragColor = OpFunctionParameter %_ptr_Function_v4float
  %fragCoord = OpFunctionParameter %_ptr_Function_v2float
         %27 = OpLabel
       %uv_0 = OpVariable %_ptr_Function_v2float Function
        %c_0 = OpVariable %_ptr_Function_float Function
      %sizer = OpVariable %_ptr_Function_float Function
     %steper = OpVariable %_ptr_Function_float Function
          %i = OpVariable %_ptr_Function_float Function
          %j = OpVariable %_ptr_Function_float Function
      %timer = OpVariable %_ptr_Function_float Function
 %resetTimer = OpVariable %_ptr_Function_float Function
    %param_1 = OpVariable %_ptr_Function_float Function
    %param_2 = OpVariable %_ptr_Function_float Function
    %param_3 = OpVariable %_ptr_Function_float Function
    %param_4 = OpVariable %_ptr_Function_v2float Function
    %param_5 = OpVariable %_ptr_Function_v2float Function
    %param_6 = OpVariable %_ptr_Function_float Function
    %param_7 = OpVariable %_ptr_Function_float Function
    %param_8 = OpVariable %_ptr_Function_float Function
    %param_9 = OpVariable %_ptr_Function_float Function
   %param_10 = OpVariable %_ptr_Function_float Function
   %param_11 = OpVariable %_ptr_Function_v2float Function
   %param_12 = OpVariable %_ptr_Function_v2float Function
   %param_13 = OpVariable %_ptr_Function_float Function
   %param_14 = OpVariable %_ptr_Function_float Function
         %61 = OpLoad %v2float %fragCoord
         %73 = OpAccessChain %_ptr_Uniform_v3float %_ %int_0
         %74 = OpLoad %v3float %73
         %75 = OpVectorShuffle %v2float %74 %74 0 1
         %76 = OpFDiv %v2float %61 %75
               OpStore %uv_0 %76
               OpStore %c_0 %float_0
         %80 = OpLoad %v2float %uv_0
         %81 = OpCompositeConstruct %v2float %float_0_5 %float_0_5
         %82 = OpFSub %v2float %80 %81
               OpStore %uv_0 %82
         %85 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %uint_0
         %86 = OpLoad %float %85
         %88 = OpAccessChain %_ptr_Uniform_float %_ %int_0 %uint_1
         %89 = OpLoad %float %88
         %90 = OpFDiv %float %86 %89
         %91 = OpAccessChain %_ptr_Function_float %uv_0 %uint_0
         %92 = OpLoad %float %91
         %93 = OpFMul %float %92 %90
         %94 = OpAccessChain %_ptr_Function_float %uv_0 %uint_0
               OpStore %94 %93
               OpStore %sizer %float_1
               OpStore %steper %float_0_100000001
        %100 = OpLoad %float %sizer
        %101 = OpFNegate %float %100
               OpStore %i %101
               OpBranch %102
        %102 = OpLabel
               OpLoopMerge %104 %105 None
               OpBranch %106
        %106 = OpLabel
        %107 = OpLoad %float %i
        %108 = OpLoad %float %sizer
        %110 = OpFOrdLessThan %bool %107 %108
               OpBranchConditional %110 %103 %104
        %103 = OpLabel
        %112 = OpLoad %float %sizer
        %113 = OpFNegate %float %112
               OpStore %j %113
               OpBranch %114
        %114 = OpLabel
               OpLoopMerge %116 %117 None
               OpBranch %118
        %118 = OpLabel
        %119 = OpLoad %float %j
        %120 = OpLoad %float %sizer
        %121 = OpFOrdLessThan %bool %119 %120
               OpBranchConditional %121 %115 %116
        %115 = OpLabel
               OpStore %timer %float_0_5
               OpStore %resetTimer %float_7
        %125 = OpLoad %float %c_0
        %126 = OpFOrdLessThanEqual %bool %125 %float_1
               OpSelectionMerge %128 None
               OpBranchConditional %126 %127 %161
        %127 = OpLabel
        %129 = OpLoad %float %i
        %130 = OpLoad %float %j
        %131 = OpCompositeConstruct %v2float %129 %130
        %133 = OpLoad %float %i
               OpStore %param_1 %133
        %134 = OpFunctionCall %float %Hash_f1_ %param_1
        %135 = OpExtInst %float %1 Sin %134
        %137 = OpLoad %float %j
               OpStore %param_2 %137
        %138 = OpFunctionCall %float %Hash_f1_ %param_2
        %139 = OpExtInst %float %1 Cos %138
        %140 = OpFMul %float %135 %139
        %142 = OpAccessChain %_ptr_Uniform_float %_ %int_1
        %143 = OpLoad %float %142
        %144 = OpLoad %float %timer
        %145 = OpFMul %float %143 %144
        %146 = OpLoad %float %resetTimer
        %147 = OpFMod %float %145 %146
        %148 = OpFMul %float %140 %147
        %150 = OpLoad %float %j
               OpStore %param_3 %150
        %151 = OpFunctionCall %float %Hash_f1_ %param_3
        %152 = OpExtInst %float %1 Sin %151
        %154 = OpLoad %v2float %uv_0
               OpStore %param_4 %154
               OpStore %param_5 %131
               OpStore %param_6 %148
               OpStore %param_7 %152
        %158 = OpFunctionCall %float %Circle_vf2_vf2_f1_f1_ %param_4 %param_5 %param_6 %param_7
        %159 = OpLoad %float %c_0
        %160 = OpFAdd %float %159 %158
               OpStore %c_0 %160
               OpBranch %128
        %161 = OpLabel
        %162 = OpLoad %float %c_0
        %163 = OpFOrdGreaterThanEqual %bool %162 %float_1
               OpSelectionMerge %165 None
               OpBranchConditional %163 %164 %165
        %164 = OpLabel
        %166 = OpLoad %float %i
        %167 = OpLoad %float %j
        %168 = OpCompositeConstruct %v2float %166 %167
        %170 = OpLoad %float %i
               OpStore %param_8 %170
        %171 = OpFunctionCall %float %Hash_f1_ %param_8
        %172 = OpExtInst %float %1 Sin %171
        %174 = OpLoad %float %j
               OpStore %param_9 %174
        %175 = OpFunctionCall %float %Hash_f1_ %param_9
        %176 = OpExtInst %float %1 Cos %175
        %177 = OpFMul %float %172 %176
        %178 = OpAccessChain %_ptr_Uniform_float %_ %int_1
        %179 = OpLoad %float %178
        %180 = OpLoad %float %timer
        %181 = OpFMul %float %179 %180
        %182 = OpLoad %float %resetTimer
        %183 = OpFMod %float %181 %182
        %184 = OpFMul %float %177 %183
        %186 = OpLoad %float %j
               OpStore %param_10 %186
        %187 = OpFunctionCall %float %Hash_f1_ %param_10
        %188 = OpExtInst %float %1 Sin %187
        %190 = OpLoad %v2float %uv_0
               OpStore %param_11 %190
               OpStore %param_12 %168
               OpStore %param_13 %184
               OpStore %param_14 %188
        %194 = OpFunctionCall %float %Circle_vf2_vf2_f1_f1_ %param_11 %param_12 %param_13 %param_14
        %195 = OpLoad %float %c_0
        %196 = OpFSub %float %195 %194
               OpStore %c_0 %196
               OpBranch %165
        %165 = OpLabel
               OpBranch %128
        %128 = OpLabel
               OpBranch %117
        %117 = OpLabel
        %197 = OpLoad %float %steper
        %198 = OpLoad %float %j
        %199 = OpFAdd %float %198 %197
               OpStore %j %199
               OpBranch %114
        %116 = OpLabel
               OpBranch %105
        %105 = OpLabel
        %200 = OpLoad %float %steper
        %201 = OpLoad %float %i
        %202 = OpFAdd %float %201 %200
               OpStore %i %202
               OpBranch %102
        %104 = OpLabel
        %203 = OpLoad %float %c_0
        %204 = OpCompositeConstruct %v3float %203 %203 %203
        %205 = OpCompositeExtract %float %204 0
        %206 = OpCompositeExtract %float %204 1
        %207 = OpCompositeExtract %float %204 2
        %208 = OpCompositeConstruct %v4float %205 %206 %207 %float_1
               OpStore %fragColor %208
               OpReturn
               OpFunctionEnd
'''