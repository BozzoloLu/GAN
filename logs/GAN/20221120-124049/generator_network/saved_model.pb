??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:@*
dtype0
?
batch_normalization_36/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_36/gamma
?
0batch_normalization_36/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_36/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_36/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_36/beta
?
/batch_normalization_36/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_36/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_36/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_36/moving_mean
?
6batch_normalization_36/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_36/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_36/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_36/moving_variance
?
:batch_normalization_36/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_36/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_transpose_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_nameconv2d_transpose_36/kernel
?
.conv2d_transpose_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_36/kernel*'
_output_shapes
:?*
dtype0
?
batch_normalization_37/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_37/gamma
?
0batch_normalization_37/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_37/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_37/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_37/beta
?
/batch_normalization_37/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_37/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_37/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_37/moving_mean
?
6batch_normalization_37/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_37/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_37/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_37/moving_variance
?
:batch_normalization_37/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_37/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*+
shared_nameconv2d_transpose_37/kernel
?
.conv2d_transpose_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_37/kernel*'
_output_shapes
:@?*
dtype0
?
batch_normalization_38/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_38/gamma
?
0batch_normalization_38/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_38/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_38/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_38/beta
?
/batch_normalization_38/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_38/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_38/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_38/moving_mean
?
6batch_normalization_38/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_38/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_38/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_38/moving_variance
?
:batch_normalization_38/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_38/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_transpose_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameconv2d_transpose_38/kernel
?
.conv2d_transpose_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_38/kernel*&
_output_shapes
:@*
dtype0

NoOpNoOp
?D
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?D
value?DB?D B?D
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
axis
	gamma
beta
moving_mean
 moving_variance
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
?

3kernel
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses*
?
:axis
	;gamma
<beta
=moving_mean
>moving_variance
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
?

Kkernel
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses*
?
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses*
?
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
?

ckernel
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses*
z
0
1
2
3
 4
35
;6
<7
=8
>9
K10
S11
T12
U13
V14
c15*
J
0
1
2
33
;4
<5
K6
S7
T8
c9*
* 
?
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

oserving_default* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_36/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_36/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_36/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_36/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
0
1
2
 3*

0
1*
* 
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 
* 
* 
jd
VARIABLE_VALUEconv2d_transpose_36/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*

30*

30*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_37/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_37/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_37/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_37/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
;0
<1
=2
>3*

;0
<1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 
* 
* 
jd
VARIABLE_VALUEconv2d_transpose_37/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*

K0*

K0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_38/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_38/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_38/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_38/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
S0
T1
U2
V3*

S0
T1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 
* 
* 
jd
VARIABLE_VALUEconv2d_transpose_38/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*

c0*

c0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*
* 
* 
.
0
 1
=2
>3
U4
V5*
Z
0
1
2
3
4
5
6
7
	8

9
10
11*
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
 1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

=0
>1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

U0
V1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
{
serving_default_input_23Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_23dense_22/kernel&batch_normalization_36/moving_variancebatch_normalization_36/gamma"batch_normalization_36/moving_meanbatch_normalization_36/betaconv2d_transpose_36/kernelbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_varianceconv2d_transpose_37/kernelbatch_normalization_38/gammabatch_normalization_38/beta"batch_normalization_38/moving_mean&batch_normalization_38/moving_varianceconv2d_transpose_38/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_796110
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_22/kernel/Read/ReadVariableOp0batch_normalization_36/gamma/Read/ReadVariableOp/batch_normalization_36/beta/Read/ReadVariableOp6batch_normalization_36/moving_mean/Read/ReadVariableOp:batch_normalization_36/moving_variance/Read/ReadVariableOp.conv2d_transpose_36/kernel/Read/ReadVariableOp0batch_normalization_37/gamma/Read/ReadVariableOp/batch_normalization_37/beta/Read/ReadVariableOp6batch_normalization_37/moving_mean/Read/ReadVariableOp:batch_normalization_37/moving_variance/Read/ReadVariableOp.conv2d_transpose_37/kernel/Read/ReadVariableOp0batch_normalization_38/gamma/Read/ReadVariableOp/batch_normalization_38/beta/Read/ReadVariableOp6batch_normalization_38/moving_mean/Read/ReadVariableOp:batch_normalization_38/moving_variance/Read/ReadVariableOp.conv2d_transpose_38/kernel/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_796560
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_22/kernelbatch_normalization_36/gammabatch_normalization_36/beta"batch_normalization_36/moving_mean&batch_normalization_36/moving_varianceconv2d_transpose_36/kernelbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_varianceconv2d_transpose_37/kernelbatch_normalization_38/gammabatch_normalization_38/beta"batch_normalization_38/moving_mean&batch_normalization_38/moving_varianceconv2d_transpose_38/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_796618??
?
?
7__inference_batch_normalization_36_layer_call_fn_796137

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_795042o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?E
?
"__inference__traced_restore_796618
file_prefix2
 assignvariableop_dense_22_kernel:@=
/assignvariableop_1_batch_normalization_36_gamma:@<
.assignvariableop_2_batch_normalization_36_beta:@C
5assignvariableop_3_batch_normalization_36_moving_mean:@G
9assignvariableop_4_batch_normalization_36_moving_variance:@H
-assignvariableop_5_conv2d_transpose_36_kernel:?>
/assignvariableop_6_batch_normalization_37_gamma:	?=
.assignvariableop_7_batch_normalization_37_beta:	?D
5assignvariableop_8_batch_normalization_37_moving_mean:	?H
9assignvariableop_9_batch_normalization_37_moving_variance:	?I
.assignvariableop_10_conv2d_transpose_37_kernel:@?>
0assignvariableop_11_batch_normalization_38_gamma:@=
/assignvariableop_12_batch_normalization_38_beta:@D
6assignvariableop_13_batch_normalization_38_moving_mean:@H
:assignvariableop_14_batch_normalization_38_moving_variance:@H
.assignvariableop_15_conv2d_transpose_38_kernel:@
identity_17??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_22_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_36_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_36_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_36_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_36_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp-assignvariableop_5_conv2d_transpose_36_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_37_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_37_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_37_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_37_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp.assignvariableop_10_conv2d_transpose_37_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_38_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_38_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp6assignvariableop_13_batch_normalization_38_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp:assignvariableop_14_batch_normalization_38_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_conv2d_transpose_38_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
f
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_795378

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????@*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
f
J__inference_leaky_re_lu_58_layer_call_and_return_conditional_losses_795432

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_795295

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
f
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_796214

inputs
identityW
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????@*
alpha%???>_
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_reshape_12_layer_call_and_return_conditional_losses_795394

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_796441

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
f
J__inference_leaky_re_lu_58_layer_call_and_return_conditional_losses_796451

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_795192

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_conv2d_transpose_37_layer_call_and_return_conditional_losses_795237

inputsC
(conv2d_transpose_readvariableop_resource:@?
identity??conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?9
?	
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795757
input_23!
dense_22_795713:@+
batch_normalization_36_795716:@+
batch_normalization_36_795718:@+
batch_normalization_36_795720:@+
batch_normalization_36_795722:@5
conv2d_transpose_36_795727:?,
batch_normalization_37_795730:	?,
batch_normalization_37_795732:	?,
batch_normalization_37_795734:	?,
batch_normalization_37_795736:	?5
conv2d_transpose_37_795740:@?+
batch_normalization_38_795743:@+
batch_normalization_38_795745:@+
batch_normalization_38_795747:@+
batch_normalization_38_795749:@4
conv2d_transpose_38_795753:@
identity??.batch_normalization_36/StatefulPartitionedCall?.batch_normalization_37/StatefulPartitionedCall?.batch_normalization_38/StatefulPartitionedCall?+conv2d_transpose_36/StatefulPartitionedCall?+conv2d_transpose_37/StatefulPartitionedCall?+conv2d_transpose_38/StatefulPartitionedCall? dense_22/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCallinput_23dense_22_795713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_795360?
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_36_795716batch_normalization_36_795718batch_normalization_36_795720batch_normalization_36_795722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_795089?
leaky_re_lu_56/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_795378?
reshape_12/PartitionedCallPartitionedCall'leaky_re_lu_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_12_layer_call_and_return_conditional_losses_795394?
+conv2d_transpose_36/StatefulPartitionedCallStatefulPartitionedCall#reshape_12/PartitionedCall:output:0conv2d_transpose_36_795727*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_36_layer_call_and_return_conditional_losses_795134?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_36/StatefulPartitionedCall:output:0batch_normalization_37_795730batch_normalization_37_795732batch_normalization_37_795734batch_normalization_37_795736*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_795192?
leaky_re_lu_57/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_57_layer_call_and_return_conditional_losses_795413?
+conv2d_transpose_37/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_57/PartitionedCall:output:0conv2d_transpose_37_795740*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_37_layer_call_and_return_conditional_losses_795237?
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_37/StatefulPartitionedCall:output:0batch_normalization_38_795743batch_normalization_38_795745batch_normalization_38_795747batch_normalization_38_795749*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_795295?
leaky_re_lu_58/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_58_layer_call_and_return_conditional_losses_795432?
+conv2d_transpose_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_58/PartitionedCall:output:0conv2d_transpose_38_795753*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_38_layer_call_and_return_conditional_losses_795341?
IdentityIdentity4conv2d_transpose_38/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall,^conv2d_transpose_36/StatefulPartitionedCall,^conv2d_transpose_37/StatefulPartitionedCall,^conv2d_transpose_38/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2Z
+conv2d_transpose_36/StatefulPartitionedCall+conv2d_transpose_36/StatefulPartitionedCall2Z
+conv2d_transpose_37/StatefulPartitionedCall+conv2d_transpose_37/StatefulPartitionedCall2Z
+conv2d_transpose_38/StatefulPartitionedCall+conv2d_transpose_38/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_23
?
b
F__inference_reshape_12_layer_call_and_return_conditional_losses_796233

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_796110
input_23
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?$
	unknown_9:@?

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@$

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_795018w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_23
?
?
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_796332

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_796314

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_38_layer_call_fn_796405

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_795295?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_795161

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_36_layer_call_fn_796150

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_795089o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_37_layer_call_fn_796283

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_795161?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?9
?	
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795591

inputs!
dense_22_795547:@+
batch_normalization_36_795550:@+
batch_normalization_36_795552:@+
batch_normalization_36_795554:@+
batch_normalization_36_795556:@5
conv2d_transpose_36_795561:?,
batch_normalization_37_795564:	?,
batch_normalization_37_795566:	?,
batch_normalization_37_795568:	?,
batch_normalization_37_795570:	?5
conv2d_transpose_37_795574:@?+
batch_normalization_38_795577:@+
batch_normalization_38_795579:@+
batch_normalization_38_795581:@+
batch_normalization_38_795583:@4
conv2d_transpose_38_795587:@
identity??.batch_normalization_36/StatefulPartitionedCall?.batch_normalization_37/StatefulPartitionedCall?.batch_normalization_38/StatefulPartitionedCall?+conv2d_transpose_36/StatefulPartitionedCall?+conv2d_transpose_37/StatefulPartitionedCall?+conv2d_transpose_38/StatefulPartitionedCall? dense_22/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCallinputsdense_22_795547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_795360?
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_36_795550batch_normalization_36_795552batch_normalization_36_795554batch_normalization_36_795556*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_795089?
leaky_re_lu_56/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_795378?
reshape_12/PartitionedCallPartitionedCall'leaky_re_lu_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_12_layer_call_and_return_conditional_losses_795394?
+conv2d_transpose_36/StatefulPartitionedCallStatefulPartitionedCall#reshape_12/PartitionedCall:output:0conv2d_transpose_36_795561*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_36_layer_call_and_return_conditional_losses_795134?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_36/StatefulPartitionedCall:output:0batch_normalization_37_795564batch_normalization_37_795566batch_normalization_37_795568batch_normalization_37_795570*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_795192?
leaky_re_lu_57/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_57_layer_call_and_return_conditional_losses_795413?
+conv2d_transpose_37/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_57/PartitionedCall:output:0conv2d_transpose_37_795574*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_37_layer_call_and_return_conditional_losses_795237?
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_37/StatefulPartitionedCall:output:0batch_normalization_38_795577batch_normalization_38_795579batch_normalization_38_795581batch_normalization_38_795583*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_795295?
leaky_re_lu_58/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_58_layer_call_and_return_conditional_losses_795432?
+conv2d_transpose_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_58/PartitionedCall:output:0conv2d_transpose_38_795587*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_38_layer_call_and_return_conditional_losses_795341?
IdentityIdentity4conv2d_transpose_38/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall,^conv2d_transpose_36/StatefulPartitionedCall,^conv2d_transpose_37/StatefulPartitionedCall,^conv2d_transpose_38/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2Z
+conv2d_transpose_36/StatefulPartitionedCall+conv2d_transpose_36/StatefulPartitionedCall2Z
+conv2d_transpose_37/StatefulPartitionedCall+conv2d_transpose_37/StatefulPartitionedCall2Z
+conv2d_transpose_38/StatefulPartitionedCall+conv2d_transpose_38/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_796423

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
2__inference_Generator_Network_layer_call_fn_795794

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?$
	unknown_9:@?

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@$

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795438w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ߒ
?
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795944

inputs9
'dense_22_matmul_readvariableop_resource:@F
8batch_normalization_36_batchnorm_readvariableop_resource:@J
<batch_normalization_36_batchnorm_mul_readvariableop_resource:@H
:batch_normalization_36_batchnorm_readvariableop_1_resource:@H
:batch_normalization_36_batchnorm_readvariableop_2_resource:@W
<conv2d_transpose_36_conv2d_transpose_readvariableop_resource:?=
.batch_normalization_37_readvariableop_resource:	??
0batch_normalization_37_readvariableop_1_resource:	?N
?batch_normalization_37_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource:	?W
<conv2d_transpose_37_conv2d_transpose_readvariableop_resource:@?<
.batch_normalization_38_readvariableop_resource:@>
0batch_normalization_38_readvariableop_1_resource:@M
?batch_normalization_38_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_38_conv2d_transpose_readvariableop_resource:@
identity??/batch_normalization_36/batchnorm/ReadVariableOp?1batch_normalization_36/batchnorm/ReadVariableOp_1?1batch_normalization_36/batchnorm/ReadVariableOp_2?3batch_normalization_36/batchnorm/mul/ReadVariableOp?6batch_normalization_37/FusedBatchNormV3/ReadVariableOp?8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_37/ReadVariableOp?'batch_normalization_37/ReadVariableOp_1?6batch_normalization_38/FusedBatchNormV3/ReadVariableOp?8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_38/ReadVariableOp?'batch_normalization_38/ReadVariableOp_1?3conv2d_transpose_36/conv2d_transpose/ReadVariableOp?3conv2d_transpose_37/conv2d_transpose/ReadVariableOp?3conv2d_transpose_38/conv2d_transpose/ReadVariableOp?dense_22/MatMul/ReadVariableOp?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0{
dense_22/MatMulMatMulinputs&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
/batch_normalization_36/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_36_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0k
&batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
$batch_normalization_36/batchnorm/addAddV27batch_normalization_36/batchnorm/ReadVariableOp:value:0/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_36/batchnorm/RsqrtRsqrt(batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:@?
3batch_normalization_36/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_36_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
$batch_normalization_36/batchnorm/mulMul*batch_normalization_36/batchnorm/Rsqrt:y:0;batch_normalization_36/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
&batch_normalization_36/batchnorm/mul_1Muldense_22/MatMul:product:0(batch_normalization_36/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
1batch_normalization_36/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_36_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_36/batchnorm/mul_2Mul9batch_normalization_36/batchnorm/ReadVariableOp_1:value:0(batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:@?
1batch_normalization_36/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_36_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0?
$batch_normalization_36/batchnorm/subSub9batch_normalization_36/batchnorm/ReadVariableOp_2:value:0*batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@?
&batch_normalization_36/batchnorm/add_1AddV2*batch_normalization_36/batchnorm/mul_1:z:0(batch_normalization_36/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
leaky_re_lu_56/LeakyRelu	LeakyRelu*batch_normalization_36/batchnorm/add_1:z:0*'
_output_shapes
:?????????@*
alpha%???>f
reshape_12/ShapeShape&leaky_re_lu_56/LeakyRelu:activations:0*
T0*
_output_shapes
:h
reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_12/strided_sliceStridedSlicereshape_12/Shape:output:0'reshape_12/strided_slice/stack:output:0)reshape_12/strided_slice/stack_1:output:0)reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_12/Reshape/shapePack!reshape_12/strided_slice:output:0#reshape_12/Reshape/shape/1:output:0#reshape_12/Reshape/shape/2:output:0#reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_12/ReshapeReshape&leaky_re_lu_56/LeakyRelu:activations:0!reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????d
conv2d_transpose_36/ShapeShapereshape_12/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_36/strided_sliceStridedSlice"conv2d_transpose_36/Shape:output:00conv2d_transpose_36/strided_slice/stack:output:02conv2d_transpose_36/strided_slice/stack_1:output:02conv2d_transpose_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_36/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_36/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_36/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
conv2d_transpose_36/stackPack*conv2d_transpose_36/strided_slice:output:0$conv2d_transpose_36/stack/1:output:0$conv2d_transpose_36/stack/2:output:0$conv2d_transpose_36/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_36/strided_slice_1StridedSlice"conv2d_transpose_36/stack:output:02conv2d_transpose_36/strided_slice_1/stack:output:04conv2d_transpose_36/strided_slice_1/stack_1:output:04conv2d_transpose_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_36/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_36_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
$conv2d_transpose_36/conv2d_transposeConv2DBackpropInput"conv2d_transpose_36/stack:output:0;conv2d_transpose_36/conv2d_transpose/ReadVariableOp:value:0reshape_12/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
%batch_normalization_37/ReadVariableOpReadVariableOp.batch_normalization_37_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_37/ReadVariableOp_1ReadVariableOp0batch_normalization_37_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_37/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_36/conv2d_transpose:output:0-batch_normalization_37/ReadVariableOp:value:0/batch_normalization_37/ReadVariableOp_1:value:0>batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
leaky_re_lu_57/LeakyRelu	LeakyRelu+batch_normalization_37/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
alpha%???>o
conv2d_transpose_37/ShapeShape&leaky_re_lu_57/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_37/strided_sliceStridedSlice"conv2d_transpose_37/Shape:output:00conv2d_transpose_37/strided_slice/stack:output:02conv2d_transpose_37/strided_slice/stack_1:output:02conv2d_transpose_37/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_37/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_37/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_37/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_37/stackPack*conv2d_transpose_37/strided_slice:output:0$conv2d_transpose_37/stack/1:output:0$conv2d_transpose_37/stack/2:output:0$conv2d_transpose_37/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_37/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_37/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_37/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_37/strided_slice_1StridedSlice"conv2d_transpose_37/stack:output:02conv2d_transpose_37/strided_slice_1/stack:output:04conv2d_transpose_37/strided_slice_1/stack_1:output:04conv2d_transpose_37/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_37/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_37_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
$conv2d_transpose_37/conv2d_transposeConv2DBackpropInput"conv2d_transpose_37/stack:output:0;conv2d_transpose_37/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_57/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
%batch_normalization_38/ReadVariableOpReadVariableOp.batch_normalization_38_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_38/ReadVariableOp_1ReadVariableOp0batch_normalization_38_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_38/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_38_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_38/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_37/conv2d_transpose:output:0-batch_normalization_38/ReadVariableOp:value:0/batch_normalization_38/ReadVariableOp_1:value:0>batch_normalization_38/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
leaky_re_lu_58/LeakyRelu	LeakyRelu+batch_normalization_38/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>o
conv2d_transpose_38/ShapeShape&leaky_re_lu_58/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_38/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_38/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_38/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_38/strided_sliceStridedSlice"conv2d_transpose_38/Shape:output:00conv2d_transpose_38/strided_slice/stack:output:02conv2d_transpose_38/strided_slice/stack_1:output:02conv2d_transpose_38/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_38/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_38/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_38/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_38/stackPack*conv2d_transpose_38/strided_slice:output:0$conv2d_transpose_38/stack/1:output:0$conv2d_transpose_38/stack/2:output:0$conv2d_transpose_38/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_38/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_38/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_38/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_38/strided_slice_1StridedSlice"conv2d_transpose_38/stack:output:02conv2d_transpose_38/strided_slice_1/stack:output:04conv2d_transpose_38/strided_slice_1/stack_1:output:04conv2d_transpose_38/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_38/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_38_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0?
$conv2d_transpose_38/conv2d_transposeConv2DBackpropInput"conv2d_transpose_38/stack:output:0;conv2d_transpose_38/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_58/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_transpose_38/TanhTanh-conv2d_transpose_38/conv2d_transpose:output:0*
T0*/
_output_shapes
:?????????s
IdentityIdentityconv2d_transpose_38/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^batch_normalization_36/batchnorm/ReadVariableOp2^batch_normalization_36/batchnorm/ReadVariableOp_12^batch_normalization_36/batchnorm/ReadVariableOp_24^batch_normalization_36/batchnorm/mul/ReadVariableOp7^batch_normalization_37/FusedBatchNormV3/ReadVariableOp9^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_37/ReadVariableOp(^batch_normalization_37/ReadVariableOp_17^batch_normalization_38/FusedBatchNormV3/ReadVariableOp9^batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_38/ReadVariableOp(^batch_normalization_38/ReadVariableOp_14^conv2d_transpose_36/conv2d_transpose/ReadVariableOp4^conv2d_transpose_37/conv2d_transpose/ReadVariableOp4^conv2d_transpose_38/conv2d_transpose/ReadVariableOp^dense_22/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2b
/batch_normalization_36/batchnorm/ReadVariableOp/batch_normalization_36/batchnorm/ReadVariableOp2f
1batch_normalization_36/batchnorm/ReadVariableOp_11batch_normalization_36/batchnorm/ReadVariableOp_12f
1batch_normalization_36/batchnorm/ReadVariableOp_21batch_normalization_36/batchnorm/ReadVariableOp_22j
3batch_normalization_36/batchnorm/mul/ReadVariableOp3batch_normalization_36/batchnorm/mul/ReadVariableOp2p
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp6batch_normalization_37/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_18batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_37/ReadVariableOp%batch_normalization_37/ReadVariableOp2R
'batch_normalization_37/ReadVariableOp_1'batch_normalization_37/ReadVariableOp_12p
6batch_normalization_38/FusedBatchNormV3/ReadVariableOp6batch_normalization_38/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_18batch_normalization_38/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_38/ReadVariableOp%batch_normalization_38/ReadVariableOp2R
'batch_normalization_38/ReadVariableOp_1'batch_normalization_38/ReadVariableOp_12j
3conv2d_transpose_36/conv2d_transpose/ReadVariableOp3conv2d_transpose_36/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_37/conv2d_transpose/ReadVariableOp3conv2d_transpose_37/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_38/conv2d_transpose/ReadVariableOp3conv2d_transpose_38/conv2d_transpose/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_796204

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
2__inference_Generator_Network_layer_call_fn_795663
input_23
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?$
	unknown_9:@?

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@$

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795591w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_23
?9
?	
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795710
input_23!
dense_22_795666:@+
batch_normalization_36_795669:@+
batch_normalization_36_795671:@+
batch_normalization_36_795673:@+
batch_normalization_36_795675:@5
conv2d_transpose_36_795680:?,
batch_normalization_37_795683:	?,
batch_normalization_37_795685:	?,
batch_normalization_37_795687:	?,
batch_normalization_37_795689:	?5
conv2d_transpose_37_795693:@?+
batch_normalization_38_795696:@+
batch_normalization_38_795698:@+
batch_normalization_38_795700:@+
batch_normalization_38_795702:@4
conv2d_transpose_38_795706:@
identity??.batch_normalization_36/StatefulPartitionedCall?.batch_normalization_37/StatefulPartitionedCall?.batch_normalization_38/StatefulPartitionedCall?+conv2d_transpose_36/StatefulPartitionedCall?+conv2d_transpose_37/StatefulPartitionedCall?+conv2d_transpose_38/StatefulPartitionedCall? dense_22/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCallinput_23dense_22_795666*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_795360?
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_36_795669batch_normalization_36_795671batch_normalization_36_795673batch_normalization_36_795675*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_795042?
leaky_re_lu_56/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_795378?
reshape_12/PartitionedCallPartitionedCall'leaky_re_lu_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_12_layer_call_and_return_conditional_losses_795394?
+conv2d_transpose_36/StatefulPartitionedCallStatefulPartitionedCall#reshape_12/PartitionedCall:output:0conv2d_transpose_36_795680*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_36_layer_call_and_return_conditional_losses_795134?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_36/StatefulPartitionedCall:output:0batch_normalization_37_795683batch_normalization_37_795685batch_normalization_37_795687batch_normalization_37_795689*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_795161?
leaky_re_lu_57/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_57_layer_call_and_return_conditional_losses_795413?
+conv2d_transpose_37/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_57/PartitionedCall:output:0conv2d_transpose_37_795693*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_37_layer_call_and_return_conditional_losses_795237?
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_37/StatefulPartitionedCall:output:0batch_normalization_38_795696batch_normalization_38_795698batch_normalization_38_795700batch_normalization_38_795702*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_795264?
leaky_re_lu_58/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_58_layer_call_and_return_conditional_losses_795432?
+conv2d_transpose_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_58/PartitionedCall:output:0conv2d_transpose_38_795706*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_38_layer_call_and_return_conditional_losses_795341?
IdentityIdentity4conv2d_transpose_38/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall,^conv2d_transpose_36/StatefulPartitionedCall,^conv2d_transpose_37/StatefulPartitionedCall,^conv2d_transpose_38/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2Z
+conv2d_transpose_36/StatefulPartitionedCall+conv2d_transpose_36/StatefulPartitionedCall2Z
+conv2d_transpose_37/StatefulPartitionedCall+conv2d_transpose_37/StatefulPartitionedCall2Z
+conv2d_transpose_38/StatefulPartitionedCall+conv2d_transpose_38/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_23
?
f
J__inference_leaky_re_lu_57_layer_call_and_return_conditional_losses_796342

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????*
alpha%???>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
O__inference_conv2d_transpose_38_layer_call_and_return_conditional_losses_795341

inputsB
(conv2d_transpose_readvariableop_resource:@
identity??conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
s
TanhTanhconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????@: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_36_layer_call_fn_796240

inputs"
unknown:?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_36_layer_call_and_return_conditional_losses_795134?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?-
?
__inference__traced_save_796560
file_prefix.
*savev2_dense_22_kernel_read_readvariableop;
7savev2_batch_normalization_36_gamma_read_readvariableop:
6savev2_batch_normalization_36_beta_read_readvariableopA
=savev2_batch_normalization_36_moving_mean_read_readvariableopE
Asavev2_batch_normalization_36_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_36_kernel_read_readvariableop;
7savev2_batch_normalization_37_gamma_read_readvariableop:
6savev2_batch_normalization_37_beta_read_readvariableopA
=savev2_batch_normalization_37_moving_mean_read_readvariableopE
Asavev2_batch_normalization_37_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_37_kernel_read_readvariableop;
7savev2_batch_normalization_38_gamma_read_readvariableop:
6savev2_batch_normalization_38_beta_read_readvariableopA
=savev2_batch_normalization_38_moving_mean_read_readvariableopE
Asavev2_batch_normalization_38_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_38_kernel_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_22_kernel_read_readvariableop7savev2_batch_normalization_36_gamma_read_readvariableop6savev2_batch_normalization_36_beta_read_readvariableop=savev2_batch_normalization_36_moving_mean_read_readvariableopAsavev2_batch_normalization_36_moving_variance_read_readvariableop5savev2_conv2d_transpose_36_kernel_read_readvariableop7savev2_batch_normalization_37_gamma_read_readvariableop6savev2_batch_normalization_37_beta_read_readvariableop=savev2_batch_normalization_37_moving_mean_read_readvariableopAsavev2_batch_normalization_37_moving_variance_read_readvariableop5savev2_conv2d_transpose_37_kernel_read_readvariableop7savev2_batch_normalization_38_gamma_read_readvariableop6savev2_batch_normalization_38_beta_read_readvariableop=savev2_batch_normalization_38_moving_mean_read_readvariableopAsavev2_batch_normalization_38_moving_variance_read_readvariableop5savev2_conv2d_transpose_38_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@:@:@:?:?:?:?:?:@?:@:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:-)
'
_output_shapes
:@?: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@:

_output_shapes
: 
??
?
!__inference__wrapped_model_795018
input_23K
9generator_network_dense_22_matmul_readvariableop_resource:@X
Jgenerator_network_batch_normalization_36_batchnorm_readvariableop_resource:@\
Ngenerator_network_batch_normalization_36_batchnorm_mul_readvariableop_resource:@Z
Lgenerator_network_batch_normalization_36_batchnorm_readvariableop_1_resource:@Z
Lgenerator_network_batch_normalization_36_batchnorm_readvariableop_2_resource:@i
Ngenerator_network_conv2d_transpose_36_conv2d_transpose_readvariableop_resource:?O
@generator_network_batch_normalization_37_readvariableop_resource:	?Q
Bgenerator_network_batch_normalization_37_readvariableop_1_resource:	?`
Qgenerator_network_batch_normalization_37_fusedbatchnormv3_readvariableop_resource:	?b
Sgenerator_network_batch_normalization_37_fusedbatchnormv3_readvariableop_1_resource:	?i
Ngenerator_network_conv2d_transpose_37_conv2d_transpose_readvariableop_resource:@?N
@generator_network_batch_normalization_38_readvariableop_resource:@P
Bgenerator_network_batch_normalization_38_readvariableop_1_resource:@_
Qgenerator_network_batch_normalization_38_fusedbatchnormv3_readvariableop_resource:@a
Sgenerator_network_batch_normalization_38_fusedbatchnormv3_readvariableop_1_resource:@h
Ngenerator_network_conv2d_transpose_38_conv2d_transpose_readvariableop_resource:@
identity??AGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp?CGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_1?CGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_2?EGenerator_Network/batch_normalization_36/batchnorm/mul/ReadVariableOp?HGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOp?JGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?7Generator_Network/batch_normalization_37/ReadVariableOp?9Generator_Network/batch_normalization_37/ReadVariableOp_1?HGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOp?JGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1?7Generator_Network/batch_normalization_38/ReadVariableOp?9Generator_Network/batch_normalization_38/ReadVariableOp_1?EGenerator_Network/conv2d_transpose_36/conv2d_transpose/ReadVariableOp?EGenerator_Network/conv2d_transpose_37/conv2d_transpose/ReadVariableOp?EGenerator_Network/conv2d_transpose_38/conv2d_transpose/ReadVariableOp?0Generator_Network/dense_22/MatMul/ReadVariableOp?
0Generator_Network/dense_22/MatMul/ReadVariableOpReadVariableOp9generator_network_dense_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
!Generator_Network/dense_22/MatMulMatMulinput_238Generator_Network/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
AGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOpReadVariableOpJgenerator_network_batch_normalization_36_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0}
8Generator_Network/batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
6Generator_Network/batch_normalization_36/batchnorm/addAddV2IGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp:value:0AGenerator_Network/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:@?
8Generator_Network/batch_normalization_36/batchnorm/RsqrtRsqrt:Generator_Network/batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:@?
EGenerator_Network/batch_normalization_36/batchnorm/mul/ReadVariableOpReadVariableOpNgenerator_network_batch_normalization_36_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
6Generator_Network/batch_normalization_36/batchnorm/mulMul<Generator_Network/batch_normalization_36/batchnorm/Rsqrt:y:0MGenerator_Network/batch_normalization_36/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
8Generator_Network/batch_normalization_36/batchnorm/mul_1Mul+Generator_Network/dense_22/MatMul:product:0:Generator_Network/batch_normalization_36/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
CGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_1ReadVariableOpLgenerator_network_batch_normalization_36_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
8Generator_Network/batch_normalization_36/batchnorm/mul_2MulKGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_1:value:0:Generator_Network/batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:@?
CGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_2ReadVariableOpLgenerator_network_batch_normalization_36_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0?
6Generator_Network/batch_normalization_36/batchnorm/subSubKGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_2:value:0<Generator_Network/batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@?
8Generator_Network/batch_normalization_36/batchnorm/add_1AddV2<Generator_Network/batch_normalization_36/batchnorm/mul_1:z:0:Generator_Network/batch_normalization_36/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
*Generator_Network/leaky_re_lu_56/LeakyRelu	LeakyRelu<Generator_Network/batch_normalization_36/batchnorm/add_1:z:0*'
_output_shapes
:?????????@*
alpha%???>?
"Generator_Network/reshape_12/ShapeShape8Generator_Network/leaky_re_lu_56/LeakyRelu:activations:0*
T0*
_output_shapes
:z
0Generator_Network/reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2Generator_Network/reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2Generator_Network/reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*Generator_Network/reshape_12/strided_sliceStridedSlice+Generator_Network/reshape_12/Shape:output:09Generator_Network/reshape_12/strided_slice/stack:output:0;Generator_Network/reshape_12/strided_slice/stack_1:output:0;Generator_Network/reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,Generator_Network/reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :n
,Generator_Network/reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :n
,Generator_Network/reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
*Generator_Network/reshape_12/Reshape/shapePack3Generator_Network/reshape_12/strided_slice:output:05Generator_Network/reshape_12/Reshape/shape/1:output:05Generator_Network/reshape_12/Reshape/shape/2:output:05Generator_Network/reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
$Generator_Network/reshape_12/ReshapeReshape8Generator_Network/leaky_re_lu_56/LeakyRelu:activations:03Generator_Network/reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
+Generator_Network/conv2d_transpose_36/ShapeShape-Generator_Network/reshape_12/Reshape:output:0*
T0*
_output_shapes
:?
9Generator_Network/conv2d_transpose_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;Generator_Network/conv2d_transpose_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;Generator_Network/conv2d_transpose_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3Generator_Network/conv2d_transpose_36/strided_sliceStridedSlice4Generator_Network/conv2d_transpose_36/Shape:output:0BGenerator_Network/conv2d_transpose_36/strided_slice/stack:output:0DGenerator_Network/conv2d_transpose_36/strided_slice/stack_1:output:0DGenerator_Network/conv2d_transpose_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-Generator_Network/conv2d_transpose_36/stack/1Const*
_output_shapes
: *
dtype0*
value	B :o
-Generator_Network/conv2d_transpose_36/stack/2Const*
_output_shapes
: *
dtype0*
value	B :p
-Generator_Network/conv2d_transpose_36/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
+Generator_Network/conv2d_transpose_36/stackPack<Generator_Network/conv2d_transpose_36/strided_slice:output:06Generator_Network/conv2d_transpose_36/stack/1:output:06Generator_Network/conv2d_transpose_36/stack/2:output:06Generator_Network/conv2d_transpose_36/stack/3:output:0*
N*
T0*
_output_shapes
:?
;Generator_Network/conv2d_transpose_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=Generator_Network/conv2d_transpose_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=Generator_Network/conv2d_transpose_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5Generator_Network/conv2d_transpose_36/strided_slice_1StridedSlice4Generator_Network/conv2d_transpose_36/stack:output:0DGenerator_Network/conv2d_transpose_36/strided_slice_1/stack:output:0FGenerator_Network/conv2d_transpose_36/strided_slice_1/stack_1:output:0FGenerator_Network/conv2d_transpose_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
EGenerator_Network/conv2d_transpose_36/conv2d_transpose/ReadVariableOpReadVariableOpNgenerator_network_conv2d_transpose_36_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
6Generator_Network/conv2d_transpose_36/conv2d_transposeConv2DBackpropInput4Generator_Network/conv2d_transpose_36/stack:output:0MGenerator_Network/conv2d_transpose_36/conv2d_transpose/ReadVariableOp:value:0-Generator_Network/reshape_12/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
7Generator_Network/batch_normalization_37/ReadVariableOpReadVariableOp@generator_network_batch_normalization_37_readvariableop_resource*
_output_shapes	
:?*
dtype0?
9Generator_Network/batch_normalization_37/ReadVariableOp_1ReadVariableOpBgenerator_network_batch_normalization_37_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
HGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOpQgenerator_network_batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
JGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpSgenerator_network_batch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
9Generator_Network/batch_normalization_37/FusedBatchNormV3FusedBatchNormV3?Generator_Network/conv2d_transpose_36/conv2d_transpose:output:0?Generator_Network/batch_normalization_37/ReadVariableOp:value:0AGenerator_Network/batch_normalization_37/ReadVariableOp_1:value:0PGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0RGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
*Generator_Network/leaky_re_lu_57/LeakyRelu	LeakyRelu=Generator_Network/batch_normalization_37/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
alpha%???>?
+Generator_Network/conv2d_transpose_37/ShapeShape8Generator_Network/leaky_re_lu_57/LeakyRelu:activations:0*
T0*
_output_shapes
:?
9Generator_Network/conv2d_transpose_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;Generator_Network/conv2d_transpose_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;Generator_Network/conv2d_transpose_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3Generator_Network/conv2d_transpose_37/strided_sliceStridedSlice4Generator_Network/conv2d_transpose_37/Shape:output:0BGenerator_Network/conv2d_transpose_37/strided_slice/stack:output:0DGenerator_Network/conv2d_transpose_37/strided_slice/stack_1:output:0DGenerator_Network/conv2d_transpose_37/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-Generator_Network/conv2d_transpose_37/stack/1Const*
_output_shapes
: *
dtype0*
value	B :o
-Generator_Network/conv2d_transpose_37/stack/2Const*
_output_shapes
: *
dtype0*
value	B :o
-Generator_Network/conv2d_transpose_37/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
+Generator_Network/conv2d_transpose_37/stackPack<Generator_Network/conv2d_transpose_37/strided_slice:output:06Generator_Network/conv2d_transpose_37/stack/1:output:06Generator_Network/conv2d_transpose_37/stack/2:output:06Generator_Network/conv2d_transpose_37/stack/3:output:0*
N*
T0*
_output_shapes
:?
;Generator_Network/conv2d_transpose_37/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=Generator_Network/conv2d_transpose_37/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=Generator_Network/conv2d_transpose_37/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5Generator_Network/conv2d_transpose_37/strided_slice_1StridedSlice4Generator_Network/conv2d_transpose_37/stack:output:0DGenerator_Network/conv2d_transpose_37/strided_slice_1/stack:output:0FGenerator_Network/conv2d_transpose_37/strided_slice_1/stack_1:output:0FGenerator_Network/conv2d_transpose_37/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
EGenerator_Network/conv2d_transpose_37/conv2d_transpose/ReadVariableOpReadVariableOpNgenerator_network_conv2d_transpose_37_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
6Generator_Network/conv2d_transpose_37/conv2d_transposeConv2DBackpropInput4Generator_Network/conv2d_transpose_37/stack:output:0MGenerator_Network/conv2d_transpose_37/conv2d_transpose/ReadVariableOp:value:08Generator_Network/leaky_re_lu_57/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
7Generator_Network/batch_normalization_38/ReadVariableOpReadVariableOp@generator_network_batch_normalization_38_readvariableop_resource*
_output_shapes
:@*
dtype0?
9Generator_Network/batch_normalization_38/ReadVariableOp_1ReadVariableOpBgenerator_network_batch_normalization_38_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
HGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOpReadVariableOpQgenerator_network_batch_normalization_38_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
JGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpSgenerator_network_batch_normalization_38_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
9Generator_Network/batch_normalization_38/FusedBatchNormV3FusedBatchNormV3?Generator_Network/conv2d_transpose_37/conv2d_transpose:output:0?Generator_Network/batch_normalization_38/ReadVariableOp:value:0AGenerator_Network/batch_normalization_38/ReadVariableOp_1:value:0PGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOp:value:0RGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
*Generator_Network/leaky_re_lu_58/LeakyRelu	LeakyRelu=Generator_Network/batch_normalization_38/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>?
+Generator_Network/conv2d_transpose_38/ShapeShape8Generator_Network/leaky_re_lu_58/LeakyRelu:activations:0*
T0*
_output_shapes
:?
9Generator_Network/conv2d_transpose_38/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;Generator_Network/conv2d_transpose_38/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;Generator_Network/conv2d_transpose_38/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3Generator_Network/conv2d_transpose_38/strided_sliceStridedSlice4Generator_Network/conv2d_transpose_38/Shape:output:0BGenerator_Network/conv2d_transpose_38/strided_slice/stack:output:0DGenerator_Network/conv2d_transpose_38/strided_slice/stack_1:output:0DGenerator_Network/conv2d_transpose_38/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-Generator_Network/conv2d_transpose_38/stack/1Const*
_output_shapes
: *
dtype0*
value	B :o
-Generator_Network/conv2d_transpose_38/stack/2Const*
_output_shapes
: *
dtype0*
value	B :o
-Generator_Network/conv2d_transpose_38/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
+Generator_Network/conv2d_transpose_38/stackPack<Generator_Network/conv2d_transpose_38/strided_slice:output:06Generator_Network/conv2d_transpose_38/stack/1:output:06Generator_Network/conv2d_transpose_38/stack/2:output:06Generator_Network/conv2d_transpose_38/stack/3:output:0*
N*
T0*
_output_shapes
:?
;Generator_Network/conv2d_transpose_38/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=Generator_Network/conv2d_transpose_38/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=Generator_Network/conv2d_transpose_38/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5Generator_Network/conv2d_transpose_38/strided_slice_1StridedSlice4Generator_Network/conv2d_transpose_38/stack:output:0DGenerator_Network/conv2d_transpose_38/strided_slice_1/stack:output:0FGenerator_Network/conv2d_transpose_38/strided_slice_1/stack_1:output:0FGenerator_Network/conv2d_transpose_38/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
EGenerator_Network/conv2d_transpose_38/conv2d_transpose/ReadVariableOpReadVariableOpNgenerator_network_conv2d_transpose_38_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0?
6Generator_Network/conv2d_transpose_38/conv2d_transposeConv2DBackpropInput4Generator_Network/conv2d_transpose_38/stack:output:0MGenerator_Network/conv2d_transpose_38/conv2d_transpose/ReadVariableOp:value:08Generator_Network/leaky_re_lu_58/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*Generator_Network/conv2d_transpose_38/TanhTanh?Generator_Network/conv2d_transpose_38/conv2d_transpose:output:0*
T0*/
_output_shapes
:??????????
IdentityIdentity.Generator_Network/conv2d_transpose_38/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????	
NoOpNoOpB^Generator_Network/batch_normalization_36/batchnorm/ReadVariableOpD^Generator_Network/batch_normalization_36/batchnorm/ReadVariableOp_1D^Generator_Network/batch_normalization_36/batchnorm/ReadVariableOp_2F^Generator_Network/batch_normalization_36/batchnorm/mul/ReadVariableOpI^Generator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOpK^Generator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_18^Generator_Network/batch_normalization_37/ReadVariableOp:^Generator_Network/batch_normalization_37/ReadVariableOp_1I^Generator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOpK^Generator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOp_18^Generator_Network/batch_normalization_38/ReadVariableOp:^Generator_Network/batch_normalization_38/ReadVariableOp_1F^Generator_Network/conv2d_transpose_36/conv2d_transpose/ReadVariableOpF^Generator_Network/conv2d_transpose_37/conv2d_transpose/ReadVariableOpF^Generator_Network/conv2d_transpose_38/conv2d_transpose/ReadVariableOp1^Generator_Network/dense_22/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2?
AGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOpAGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp2?
CGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_1CGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_12?
CGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_2CGenerator_Network/batch_normalization_36/batchnorm/ReadVariableOp_22?
EGenerator_Network/batch_normalization_36/batchnorm/mul/ReadVariableOpEGenerator_Network/batch_normalization_36/batchnorm/mul/ReadVariableOp2?
HGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOpHGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOp2?
JGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1JGenerator_Network/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12r
7Generator_Network/batch_normalization_37/ReadVariableOp7Generator_Network/batch_normalization_37/ReadVariableOp2v
9Generator_Network/batch_normalization_37/ReadVariableOp_19Generator_Network/batch_normalization_37/ReadVariableOp_12?
HGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOpHGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOp2?
JGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1JGenerator_Network/batch_normalization_38/FusedBatchNormV3/ReadVariableOp_12r
7Generator_Network/batch_normalization_38/ReadVariableOp7Generator_Network/batch_normalization_38/ReadVariableOp2v
9Generator_Network/batch_normalization_38/ReadVariableOp_19Generator_Network/batch_normalization_38/ReadVariableOp_12?
EGenerator_Network/conv2d_transpose_36/conv2d_transpose/ReadVariableOpEGenerator_Network/conv2d_transpose_36/conv2d_transpose/ReadVariableOp2?
EGenerator_Network/conv2d_transpose_37/conv2d_transpose/ReadVariableOpEGenerator_Network/conv2d_transpose_37/conv2d_transpose/ReadVariableOp2?
EGenerator_Network/conv2d_transpose_38/conv2d_transpose/ReadVariableOpEGenerator_Network/conv2d_transpose_38/conv2d_transpose/ReadVariableOp2d
0Generator_Network/dense_22/MatMul/ReadVariableOp0Generator_Network/dense_22/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_23
?
?
2__inference_Generator_Network_layer_call_fn_795831

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?$
	unknown_9:@?

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@$

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795591w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_795042

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
O__inference_conv2d_transpose_38_layer_call_and_return_conditional_losses_796489

inputsB
(conv2d_transpose_readvariableop_resource:@
identity??conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
s
TanhTanhconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????@: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?9
?	
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795438

inputs!
dense_22_795361:@+
batch_normalization_36_795364:@+
batch_normalization_36_795366:@+
batch_normalization_36_795368:@+
batch_normalization_36_795370:@5
conv2d_transpose_36_795396:?,
batch_normalization_37_795399:	?,
batch_normalization_37_795401:	?,
batch_normalization_37_795403:	?,
batch_normalization_37_795405:	?5
conv2d_transpose_37_795415:@?+
batch_normalization_38_795418:@+
batch_normalization_38_795420:@+
batch_normalization_38_795422:@+
batch_normalization_38_795424:@4
conv2d_transpose_38_795434:@
identity??.batch_normalization_36/StatefulPartitionedCall?.batch_normalization_37/StatefulPartitionedCall?.batch_normalization_38/StatefulPartitionedCall?+conv2d_transpose_36/StatefulPartitionedCall?+conv2d_transpose_37/StatefulPartitionedCall?+conv2d_transpose_38/StatefulPartitionedCall? dense_22/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCallinputsdense_22_795361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_795360?
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0batch_normalization_36_795364batch_normalization_36_795366batch_normalization_36_795368batch_normalization_36_795370*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_795042?
leaky_re_lu_56/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_795378?
reshape_12/PartitionedCallPartitionedCall'leaky_re_lu_56/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_12_layer_call_and_return_conditional_losses_795394?
+conv2d_transpose_36/StatefulPartitionedCallStatefulPartitionedCall#reshape_12/PartitionedCall:output:0conv2d_transpose_36_795396*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_36_layer_call_and_return_conditional_losses_795134?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_36/StatefulPartitionedCall:output:0batch_normalization_37_795399batch_normalization_37_795401batch_normalization_37_795403batch_normalization_37_795405*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_795161?
leaky_re_lu_57/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_57_layer_call_and_return_conditional_losses_795413?
+conv2d_transpose_37/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_57/PartitionedCall:output:0conv2d_transpose_37_795415*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_37_layer_call_and_return_conditional_losses_795237?
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_37/StatefulPartitionedCall:output:0batch_normalization_38_795418batch_normalization_38_795420batch_normalization_38_795422batch_normalization_38_795424*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_795264?
leaky_re_lu_58/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_58_layer_call_and_return_conditional_losses_795432?
+conv2d_transpose_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_58/PartitionedCall:output:0conv2d_transpose_38_795434*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_38_layer_call_and_return_conditional_losses_795341?
IdentityIdentity4conv2d_transpose_38/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall,^conv2d_transpose_36/StatefulPartitionedCall,^conv2d_transpose_37/StatefulPartitionedCall,^conv2d_transpose_38/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2Z
+conv2d_transpose_36/StatefulPartitionedCall+conv2d_transpose_36/StatefulPartitionedCall2Z
+conv2d_transpose_37/StatefulPartitionedCall+conv2d_transpose_37/StatefulPartitionedCall2Z
+conv2d_transpose_38/StatefulPartitionedCall+conv2d_transpose_38/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_38_layer_call_fn_796392

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_795264?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_38_layer_call_fn_796458

inputs!
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_38_layer_call_and_return_conditional_losses_795341?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????@: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_conv2d_transpose_36_layer_call_and_return_conditional_losses_795134

inputsC
(conv2d_transpose_readvariableop_resource:?
identity??conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_796170

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?%
?
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_795089

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_37_layer_call_fn_796349

inputs"
unknown:@?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_conv2d_transpose_37_layer_call_and_return_conditional_losses_795237?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_Generator_Network_layer_call_fn_795473
input_23
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?$
	unknown_9:@?

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@$

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795438w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_23
?
?
O__inference_conv2d_transpose_36_layer_call_and_return_conditional_losses_796270

inputsC
(conv2d_transpose_readvariableop_resource:?
identity??conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+???????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_reshape_12_layer_call_fn_796219

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_reshape_12_layer_call_and_return_conditional_losses_795394h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
f
J__inference_leaky_re_lu_57_layer_call_and_return_conditional_losses_795413

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????*
alpha%???>h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_795264

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
պ
?
M__inference_Generator_Network_layer_call_and_return_conditional_losses_796071

inputs9
'dense_22_matmul_readvariableop_resource:@L
>batch_normalization_36_assignmovingavg_readvariableop_resource:@N
@batch_normalization_36_assignmovingavg_1_readvariableop_resource:@J
<batch_normalization_36_batchnorm_mul_readvariableop_resource:@F
8batch_normalization_36_batchnorm_readvariableop_resource:@W
<conv2d_transpose_36_conv2d_transpose_readvariableop_resource:?=
.batch_normalization_37_readvariableop_resource:	??
0batch_normalization_37_readvariableop_1_resource:	?N
?batch_normalization_37_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource:	?W
<conv2d_transpose_37_conv2d_transpose_readvariableop_resource:@?<
.batch_normalization_38_readvariableop_resource:@>
0batch_normalization_38_readvariableop_1_resource:@M
?batch_normalization_38_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_38_conv2d_transpose_readvariableop_resource:@
identity??&batch_normalization_36/AssignMovingAvg?5batch_normalization_36/AssignMovingAvg/ReadVariableOp?(batch_normalization_36/AssignMovingAvg_1?7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_36/batchnorm/ReadVariableOp?3batch_normalization_36/batchnorm/mul/ReadVariableOp?%batch_normalization_37/AssignNewValue?'batch_normalization_37/AssignNewValue_1?6batch_normalization_37/FusedBatchNormV3/ReadVariableOp?8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_37/ReadVariableOp?'batch_normalization_37/ReadVariableOp_1?%batch_normalization_38/AssignNewValue?'batch_normalization_38/AssignNewValue_1?6batch_normalization_38/FusedBatchNormV3/ReadVariableOp?8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_38/ReadVariableOp?'batch_normalization_38/ReadVariableOp_1?3conv2d_transpose_36/conv2d_transpose/ReadVariableOp?3conv2d_transpose_37/conv2d_transpose/ReadVariableOp?3conv2d_transpose_38/conv2d_transpose/ReadVariableOp?dense_22/MatMul/ReadVariableOp?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0{
dense_22/MatMulMatMulinputs&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@
5batch_normalization_36/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
#batch_normalization_36/moments/meanMeandense_22/MatMul:product:0>batch_normalization_36/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(?
+batch_normalization_36/moments/StopGradientStopGradient,batch_normalization_36/moments/mean:output:0*
T0*
_output_shapes

:@?
0batch_normalization_36/moments/SquaredDifferenceSquaredDifferencedense_22/MatMul:product:04batch_normalization_36/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@?
9batch_normalization_36/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
'batch_normalization_36/moments/varianceMean4batch_normalization_36/moments/SquaredDifference:z:0Bbatch_normalization_36/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(?
&batch_normalization_36/moments/SqueezeSqueeze,batch_normalization_36/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 ?
(batch_normalization_36/moments/Squeeze_1Squeeze0batch_normalization_36/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 q
,batch_normalization_36/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
5batch_normalization_36/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_36_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0?
*batch_normalization_36/AssignMovingAvg/subSub=batch_normalization_36/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_36/moments/Squeeze:output:0*
T0*
_output_shapes
:@?
*batch_normalization_36/AssignMovingAvg/mulMul.batch_normalization_36/AssignMovingAvg/sub:z:05batch_normalization_36/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@?
&batch_normalization_36/AssignMovingAvgAssignSubVariableOp>batch_normalization_36_assignmovingavg_readvariableop_resource.batch_normalization_36/AssignMovingAvg/mul:z:06^batch_normalization_36/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_36/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_36_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
,batch_normalization_36/AssignMovingAvg_1/subSub?batch_normalization_36/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_36/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@?
,batch_normalization_36/AssignMovingAvg_1/mulMul0batch_normalization_36/AssignMovingAvg_1/sub:z:07batch_normalization_36/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@?
(batch_normalization_36/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_36_assignmovingavg_1_readvariableop_resource0batch_normalization_36/AssignMovingAvg_1/mul:z:08^batch_normalization_36/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_36/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
$batch_normalization_36/batchnorm/addAddV21batch_normalization_36/moments/Squeeze_1:output:0/batch_normalization_36/batchnorm/add/y:output:0*
T0*
_output_shapes
:@~
&batch_normalization_36/batchnorm/RsqrtRsqrt(batch_normalization_36/batchnorm/add:z:0*
T0*
_output_shapes
:@?
3batch_normalization_36/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_36_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0?
$batch_normalization_36/batchnorm/mulMul*batch_normalization_36/batchnorm/Rsqrt:y:0;batch_normalization_36/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@?
&batch_normalization_36/batchnorm/mul_1Muldense_22/MatMul:product:0(batch_normalization_36/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@?
&batch_normalization_36/batchnorm/mul_2Mul/batch_normalization_36/moments/Squeeze:output:0(batch_normalization_36/batchnorm/mul:z:0*
T0*
_output_shapes
:@?
/batch_normalization_36/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_36_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0?
$batch_normalization_36/batchnorm/subSub7batch_normalization_36/batchnorm/ReadVariableOp:value:0*batch_normalization_36/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@?
&batch_normalization_36/batchnorm/add_1AddV2*batch_normalization_36/batchnorm/mul_1:z:0(batch_normalization_36/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@?
leaky_re_lu_56/LeakyRelu	LeakyRelu*batch_normalization_36/batchnorm/add_1:z:0*'
_output_shapes
:?????????@*
alpha%???>f
reshape_12/ShapeShape&leaky_re_lu_56/LeakyRelu:activations:0*
T0*
_output_shapes
:h
reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_12/strided_sliceStridedSlicereshape_12/Shape:output:0'reshape_12/strided_slice/stack:output:0)reshape_12/strided_slice/stack_1:output:0)reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_12/Reshape/shapePack!reshape_12/strided_slice:output:0#reshape_12/Reshape/shape/1:output:0#reshape_12/Reshape/shape/2:output:0#reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_12/ReshapeReshape&leaky_re_lu_56/LeakyRelu:activations:0!reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????d
conv2d_transpose_36/ShapeShapereshape_12/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_36/strided_sliceStridedSlice"conv2d_transpose_36/Shape:output:00conv2d_transpose_36/strided_slice/stack:output:02conv2d_transpose_36/strided_slice/stack_1:output:02conv2d_transpose_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_36/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_36/stack/2Const*
_output_shapes
: *
dtype0*
value	B :^
conv2d_transpose_36/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
conv2d_transpose_36/stackPack*conv2d_transpose_36/strided_slice:output:0$conv2d_transpose_36/stack/1:output:0$conv2d_transpose_36/stack/2:output:0$conv2d_transpose_36/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_36/strided_slice_1StridedSlice"conv2d_transpose_36/stack:output:02conv2d_transpose_36/strided_slice_1/stack:output:04conv2d_transpose_36/strided_slice_1/stack_1:output:04conv2d_transpose_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_36/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_36_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype0?
$conv2d_transpose_36/conv2d_transposeConv2DBackpropInput"conv2d_transpose_36/stack:output:0;conv2d_transpose_36/conv2d_transpose/ReadVariableOp:value:0reshape_12/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
%batch_normalization_37/ReadVariableOpReadVariableOp.batch_normalization_37_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_37/ReadVariableOp_1ReadVariableOp0batch_normalization_37_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_37/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_36/conv2d_transpose:output:0-batch_normalization_37/ReadVariableOp:value:0/batch_normalization_37/ReadVariableOp_1:value:0>batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_37/AssignNewValueAssignVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource4batch_normalization_37/FusedBatchNormV3:batch_mean:07^batch_normalization_37/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_37/AssignNewValue_1AssignVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_37/FusedBatchNormV3:batch_variance:09^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_57/LeakyRelu	LeakyRelu+batch_normalization_37/FusedBatchNormV3:y:0*0
_output_shapes
:??????????*
alpha%???>o
conv2d_transpose_37/ShapeShape&leaky_re_lu_57/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_37/strided_sliceStridedSlice"conv2d_transpose_37/Shape:output:00conv2d_transpose_37/strided_slice/stack:output:02conv2d_transpose_37/strided_slice/stack_1:output:02conv2d_transpose_37/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_37/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_37/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_37/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_37/stackPack*conv2d_transpose_37/strided_slice:output:0$conv2d_transpose_37/stack/1:output:0$conv2d_transpose_37/stack/2:output:0$conv2d_transpose_37/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_37/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_37/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_37/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_37/strided_slice_1StridedSlice"conv2d_transpose_37/stack:output:02conv2d_transpose_37/strided_slice_1/stack:output:04conv2d_transpose_37/strided_slice_1/stack_1:output:04conv2d_transpose_37/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_37/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_37_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
$conv2d_transpose_37/conv2d_transposeConv2DBackpropInput"conv2d_transpose_37/stack:output:0;conv2d_transpose_37/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_57/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
%batch_normalization_38/ReadVariableOpReadVariableOp.batch_normalization_38_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_38/ReadVariableOp_1ReadVariableOp0batch_normalization_38_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_38/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_38_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_38/FusedBatchNormV3FusedBatchNormV3-conv2d_transpose_37/conv2d_transpose:output:0-batch_normalization_38/ReadVariableOp:value:0/batch_normalization_38/ReadVariableOp_1:value:0>batch_normalization_38/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_38/AssignNewValueAssignVariableOp?batch_normalization_38_fusedbatchnormv3_readvariableop_resource4batch_normalization_38/FusedBatchNormV3:batch_mean:07^batch_normalization_38/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_38/AssignNewValue_1AssignVariableOpAbatch_normalization_38_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_38/FusedBatchNormV3:batch_variance:09^batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_58/LeakyRelu	LeakyRelu+batch_normalization_38/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@*
alpha%???>o
conv2d_transpose_38/ShapeShape&leaky_re_lu_58/LeakyRelu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_38/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_38/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_38/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_38/strided_sliceStridedSlice"conv2d_transpose_38/Shape:output:00conv2d_transpose_38/strided_slice/stack:output:02conv2d_transpose_38/strided_slice/stack_1:output:02conv2d_transpose_38/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_38/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_38/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_38/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_38/stackPack*conv2d_transpose_38/strided_slice:output:0$conv2d_transpose_38/stack/1:output:0$conv2d_transpose_38/stack/2:output:0$conv2d_transpose_38/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_38/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_38/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_38/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_38/strided_slice_1StridedSlice"conv2d_transpose_38/stack:output:02conv2d_transpose_38/strided_slice_1/stack:output:04conv2d_transpose_38/strided_slice_1/stack_1:output:04conv2d_transpose_38/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_38/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_38_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0?
$conv2d_transpose_38/conv2d_transposeConv2DBackpropInput"conv2d_transpose_38/stack:output:0;conv2d_transpose_38/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_58/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_transpose_38/TanhTanh-conv2d_transpose_38/conv2d_transpose:output:0*
T0*/
_output_shapes
:?????????s
IdentityIdentityconv2d_transpose_38/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp'^batch_normalization_36/AssignMovingAvg6^batch_normalization_36/AssignMovingAvg/ReadVariableOp)^batch_normalization_36/AssignMovingAvg_18^batch_normalization_36/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_36/batchnorm/ReadVariableOp4^batch_normalization_36/batchnorm/mul/ReadVariableOp&^batch_normalization_37/AssignNewValue(^batch_normalization_37/AssignNewValue_17^batch_normalization_37/FusedBatchNormV3/ReadVariableOp9^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_37/ReadVariableOp(^batch_normalization_37/ReadVariableOp_1&^batch_normalization_38/AssignNewValue(^batch_normalization_38/AssignNewValue_17^batch_normalization_38/FusedBatchNormV3/ReadVariableOp9^batch_normalization_38/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_38/ReadVariableOp(^batch_normalization_38/ReadVariableOp_14^conv2d_transpose_36/conv2d_transpose/ReadVariableOp4^conv2d_transpose_37/conv2d_transpose/ReadVariableOp4^conv2d_transpose_38/conv2d_transpose/ReadVariableOp^dense_22/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : : : : : 2P
&batch_normalization_36/AssignMovingAvg&batch_normalization_36/AssignMovingAvg2n
5batch_normalization_36/AssignMovingAvg/ReadVariableOp5batch_normalization_36/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_36/AssignMovingAvg_1(batch_normalization_36/AssignMovingAvg_12r
7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp7batch_normalization_36/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_36/batchnorm/ReadVariableOp/batch_normalization_36/batchnorm/ReadVariableOp2j
3batch_normalization_36/batchnorm/mul/ReadVariableOp3batch_normalization_36/batchnorm/mul/ReadVariableOp2N
%batch_normalization_37/AssignNewValue%batch_normalization_37/AssignNewValue2R
'batch_normalization_37/AssignNewValue_1'batch_normalization_37/AssignNewValue_12p
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp6batch_normalization_37/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_18batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_37/ReadVariableOp%batch_normalization_37/ReadVariableOp2R
'batch_normalization_37/ReadVariableOp_1'batch_normalization_37/ReadVariableOp_12N
%batch_normalization_38/AssignNewValue%batch_normalization_38/AssignNewValue2R
'batch_normalization_38/AssignNewValue_1'batch_normalization_38/AssignNewValue_12p
6batch_normalization_38/FusedBatchNormV3/ReadVariableOp6batch_normalization_38/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_38/FusedBatchNormV3/ReadVariableOp_18batch_normalization_38/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_38/ReadVariableOp%batch_normalization_38/ReadVariableOp2R
'batch_normalization_38/ReadVariableOp_1'batch_normalization_38/ReadVariableOp_12j
3conv2d_transpose_36/conv2d_transpose/ReadVariableOp3conv2d_transpose_36/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_37/conv2d_transpose/ReadVariableOp3conv2d_transpose_37/conv2d_transpose/ReadVariableOp2j
3conv2d_transpose_38/conv2d_transpose/ReadVariableOp3conv2d_transpose_38/conv2d_transpose/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_dense_22_layer_call_and_return_conditional_losses_796124

inputs0
matmul_readvariableop_resource:@
identity??MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_leaky_re_lu_56_layer_call_fn_796209

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_795378`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
O__inference_conv2d_transpose_37_layer_call_and_return_conditional_losses_796379

inputsC
(conv2d_transpose_readvariableop_resource:@?
identity??conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
}
)__inference_dense_22_layer_call_fn_796117

inputs
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_795360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_dense_22_layer_call_and_return_conditional_losses_795360

inputs0
matmul_readvariableop_resource:@
identity??MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_37_layer_call_fn_796296

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_795192?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_leaky_re_lu_57_layer_call_fn_796337

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_57_layer_call_and_return_conditional_losses_795413i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_leaky_re_lu_58_layer_call_fn_796446

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_leaky_re_lu_58_layer_call_and_return_conditional_losses_795432h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_231
serving_default_input_23:0?????????O
conv2d_transpose_388
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	gamma
beta
moving_mean
 moving_variance
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
?

3kernel
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
?
:axis
	;gamma
<beta
=moving_mean
>moving_variance
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Kkernel
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
?
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ckernel
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0
1
2
3
 4
35
;6
<7
=8
>9
K10
S11
T12
U13
V14
c15"
trackable_list_wrapper
f
0
1
2
33
;4
<5
K6
S7
T8
c9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_Generator_Network_layer_call_fn_795473
2__inference_Generator_Network_layer_call_fn_795794
2__inference_Generator_Network_layer_call_fn_795831
2__inference_Generator_Network_layer_call_fn_795663?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795944
M__inference_Generator_Network_layer_call_and_return_conditional_losses_796071
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795710
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795757?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_795018input_23"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
oserving_default"
signature_map
!:@2dense_22/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_22_layer_call_fn_796117?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_22_layer_call_and_return_conditional_losses_796124?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
*:(@2batch_normalization_36/gamma
):'@2batch_normalization_36/beta
2:0@ (2"batch_normalization_36/moving_mean
6:4@ (2&batch_normalization_36/moving_variance
<
0
1
2
 3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
?2?
7__inference_batch_normalization_36_layer_call_fn_796137
7__inference_batch_normalization_36_layer_call_fn_796150?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_796170
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_796204?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_leaky_re_lu_56_layer_call_fn_796209?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_796214?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_reshape_12_layer_call_fn_796219?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_reshape_12_layer_call_and_return_conditional_losses_796233?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
5:3?2conv2d_transpose_36/kernel
'
30"
trackable_list_wrapper
'
30"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
?2?
4__inference_conv2d_transpose_36_layer_call_fn_796240?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_36_layer_call_and_return_conditional_losses_796270?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
+:)?2batch_normalization_37/gamma
*:(?2batch_normalization_37/beta
3:1? (2"batch_normalization_37/moving_mean
7:5? (2&batch_normalization_37/moving_variance
<
;0
<1
=2
>3"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
?2?
7__inference_batch_normalization_37_layer_call_fn_796283
7__inference_batch_normalization_37_layer_call_fn_796296?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_796314
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_796332?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_leaky_re_lu_57_layer_call_fn_796337?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_leaky_re_lu_57_layer_call_and_return_conditional_losses_796342?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
5:3@?2conv2d_transpose_37/kernel
'
K0"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
?2?
4__inference_conv2d_transpose_37_layer_call_fn_796349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_37_layer_call_and_return_conditional_losses_796379?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
*:(@2batch_normalization_38/gamma
):'@2batch_normalization_38/beta
2:0@ (2"batch_normalization_38/moving_mean
6:4@ (2&batch_normalization_38/moving_variance
<
S0
T1
U2
V3"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
?2?
7__inference_batch_normalization_38_layer_call_fn_796392
7__inference_batch_normalization_38_layer_call_fn_796405?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_796423
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_796441?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_leaky_re_lu_58_layer_call_fn_796446?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_leaky_re_lu_58_layer_call_and_return_conditional_losses_796451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
4:2@2conv2d_transpose_38/kernel
'
c0"
trackable_list_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
?2?
4__inference_conv2d_transpose_38_layer_call_fn_796458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_conv2d_transpose_38_layer_call_and_return_conditional_losses_796489?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
J
0
 1
=2
>3
U4
V5"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_796110input_23"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper?
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795710| 3;<=>KSTUVc9?6
/?,
"?
input_23?????????
p 

 
? "-?*
#? 
0?????????
? ?
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795757| 3;<=>KSTUVc9?6
/?,
"?
input_23?????????
p

 
? "-?*
#? 
0?????????
? ?
M__inference_Generator_Network_layer_call_and_return_conditional_losses_795944z 3;<=>KSTUVc7?4
-?*
 ?
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
M__inference_Generator_Network_layer_call_and_return_conditional_losses_796071z 3;<=>KSTUVc7?4
-?*
 ?
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
2__inference_Generator_Network_layer_call_fn_795473o 3;<=>KSTUVc9?6
/?,
"?
input_23?????????
p 

 
? " ???????????
2__inference_Generator_Network_layer_call_fn_795663o 3;<=>KSTUVc9?6
/?,
"?
input_23?????????
p

 
? " ???????????
2__inference_Generator_Network_layer_call_fn_795794m 3;<=>KSTUVc7?4
-?*
 ?
inputs?????????
p 

 
? " ???????????
2__inference_Generator_Network_layer_call_fn_795831m 3;<=>KSTUVc7?4
-?*
 ?
inputs?????????
p

 
? " ???????????
!__inference__wrapped_model_795018? 3;<=>KSTUVc1?.
'?$
"?
input_23?????????
? "Q?N
L
conv2d_transpose_385?2
conv2d_transpose_38??????????
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_796170b 3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
R__inference_batch_normalization_36_layer_call_and_return_conditional_losses_796204b 3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
7__inference_batch_normalization_36_layer_call_fn_796137U 3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
7__inference_batch_normalization_36_layer_call_fn_796150U 3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_796314?;<=>N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_37_layer_call_and_return_conditional_losses_796332?;<=>N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
7__inference_batch_normalization_37_layer_call_fn_796283?;<=>N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
7__inference_batch_normalization_37_layer_call_fn_796296?;<=>N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_796423?STUVM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_38_layer_call_and_return_conditional_losses_796441?STUVM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
7__inference_batch_normalization_38_layer_call_fn_796392?STUVM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
7__inference_batch_normalization_38_layer_call_fn_796405?STUVM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
O__inference_conv2d_transpose_36_layer_call_and_return_conditional_losses_796270?3I?F
??<
:?7
inputs+???????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
4__inference_conv2d_transpose_36_layer_call_fn_796240?3I?F
??<
:?7
inputs+???????????????????????????
? "3?0,?????????????????????????????
O__inference_conv2d_transpose_37_layer_call_and_return_conditional_losses_796379?KJ?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
4__inference_conv2d_transpose_37_layer_call_fn_796349?KJ?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
O__inference_conv2d_transpose_38_layer_call_and_return_conditional_losses_796489?cI?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????
? ?
4__inference_conv2d_transpose_38_layer_call_fn_796458?cI?F
??<
:?7
inputs+???????????????????????????@
? "2?/+????????????????????????????
D__inference_dense_22_layer_call_and_return_conditional_losses_796124[/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? {
)__inference_dense_22_layer_call_fn_796117N/?,
%?"
 ?
inputs?????????
? "??????????@?
J__inference_leaky_re_lu_56_layer_call_and_return_conditional_losses_796214X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
/__inference_leaky_re_lu_56_layer_call_fn_796209K/?,
%?"
 ?
inputs?????????@
? "??????????@?
J__inference_leaky_re_lu_57_layer_call_and_return_conditional_losses_796342j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
/__inference_leaky_re_lu_57_layer_call_fn_796337]8?5
.?+
)?&
inputs??????????
? "!????????????
J__inference_leaky_re_lu_58_layer_call_and_return_conditional_losses_796451h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
/__inference_leaky_re_lu_58_layer_call_fn_796446[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
F__inference_reshape_12_layer_call_and_return_conditional_losses_796233`/?,
%?"
 ?
inputs?????????@
? "-?*
#? 
0?????????
? ?
+__inference_reshape_12_layer_call_fn_796219S/?,
%?"
 ?
inputs?????????@
? " ???????????
$__inference_signature_wrapper_796110? 3;<=>KSTUVc=?:
? 
3?0
.
input_23"?
input_23?????????"Q?N
L
conv2d_transpose_385?2
conv2d_transpose_38?????????