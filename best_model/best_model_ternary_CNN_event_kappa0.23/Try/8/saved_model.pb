кн
иў
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
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
√
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
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58╒╧
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
А
Adam/v/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_26/bias
y
(Adam/v/dense_26/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_26/bias*
_output_shapes
:*
dtype0
А
Adam/m/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_26/bias
y
(Adam/m/dense_26/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_26/bias*
_output_shapes
:*
dtype0
Й
Adam/v/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/v/dense_26/kernel
В
*Adam/v/dense_26/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_26/kernel*
_output_shapes
:	А*
dtype0
Й
Adam/m/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/m/dense_26/kernel
В
*Adam/m/dense_26/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_26/kernel*
_output_shapes
:	А*
dtype0
Б
Adam/v/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/dense_25/bias
z
(Adam/v/dense_25/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_25/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/dense_25/bias
z
(Adam/m/dense_25/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_25/bias*
_output_shapes	
:А*
dtype0
К
Adam/v/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/v/dense_25/kernel
Г
*Adam/v/dense_25/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_25/kernel* 
_output_shapes
:
АА*
dtype0
К
Adam/m/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/m/dense_25/kernel
Г
*Adam/m/dense_25/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_25/kernel* 
_output_shapes
:
АА*
dtype0
Б
Adam/v/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/dense_24/bias
z
(Adam/v/dense_24/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_24/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/dense_24/bias
z
(Adam/m/dense_24/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_24/bias*
_output_shapes	
:А*
dtype0
Л
Adam/v/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*'
shared_nameAdam/v/dense_24/kernel
Д
*Adam/v/dense_24/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_24/kernel*!
_output_shapes
:АвА*
dtype0
Л
Adam/m/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*'
shared_nameAdam/m/dense_24/kernel
Д
*Adam/m/dense_24/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_24/kernel*!
_output_shapes
:АвА*
dtype0
Г
Adam/v/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/v/conv2d_26/bias
|
)Adam/v/conv2d_26/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_26/bias*
_output_shapes	
:А*
dtype0
Г
Adam/m/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/m/conv2d_26/bias
|
)Adam/m/conv2d_26/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_26/bias*
_output_shapes	
:А*
dtype0
Ф
Adam/v/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/v/conv2d_26/kernel
Н
+Adam/v/conv2d_26/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_26/kernel*(
_output_shapes
:АА*
dtype0
Ф
Adam/m/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/m/conv2d_26/kernel
Н
+Adam/m/conv2d_26/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_26/kernel*(
_output_shapes
:АА*
dtype0
Г
Adam/v/conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/v/conv2d_25/bias
|
)Adam/v/conv2d_25/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_25/bias*
_output_shapes	
:А*
dtype0
Г
Adam/m/conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/m/conv2d_25/bias
|
)Adam/m/conv2d_25/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_25/bias*
_output_shapes	
:А*
dtype0
У
Adam/v/conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/v/conv2d_25/kernel
М
+Adam/v/conv2d_25/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_25/kernel*'
_output_shapes
: А*
dtype0
У
Adam/m/conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/m/conv2d_25/kernel
М
+Adam/m/conv2d_25/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_25/kernel*'
_output_shapes
: А*
dtype0
В
Adam/v/conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/conv2d_24/bias
{
)Adam/v/conv2d_24/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_24/bias*
_output_shapes
: *
dtype0
В
Adam/m/conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/conv2d_24/bias
{
)Adam/m/conv2d_24/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_24/bias*
_output_shapes
: *
dtype0
Т
Adam/v/conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/v/conv2d_24/kernel
Л
+Adam/v/conv2d_24/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_24/kernel*&
_output_shapes
: *
dtype0
Т
Adam/m/conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/m/conv2d_24/kernel
Л
+Adam/m/conv2d_24/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_24/kernel*&
_output_shapes
: *
dtype0
Ъ
!Adam/v/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/batch_normalization_8/beta
У
5Adam/v/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_8/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/batch_normalization_8/beta
У
5Adam/m/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_8/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_8/gamma
Х
6Adam/v/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_8/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_8/gamma
Х
6Adam/m/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_8/gamma*
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:*
dtype0
{
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_26/kernel
t
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes
:	А*
dtype0
s
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_25/bias
l
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes	
:А*
dtype0
|
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_25/kernel
u
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_24/bias
l
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes	
:А*
dtype0
}
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА* 
shared_namedense_24/kernel
v
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*!
_output_shapes
:АвА*
dtype0
u
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_26/bias
n
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_26/kernel

$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_25/bias
n
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes	
:А*
dtype0
Е
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*!
shared_nameconv2d_25/kernel
~
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*'
_output_shapes
: А*
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
: *
dtype0
Д
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
: *
dtype0
в
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance
Ы
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean
У
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
М
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta
Е
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0
О
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma
З
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0
К
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
Ь
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_2179944

NoOpNoOp
АВ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*║Б
valueпБBлБ BгБ
я
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

h2ptjl
	_output

	optimizer
call

signatures*
z
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15*
j
0
1
2
3
4
5
6
7
8
9
10
11
12
13*
* 
░
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
"trace_0
#trace_1
$trace_2
%trace_3* 
6
&trace_0
'trace_1
(trace_2
)trace_3* 
* 
ц
*layer-0
+layer_with_weights-0
+layer-1
,layer_with_weights-1
,layer-2
-layer-3
.layer_with_weights-2
.layer-4
/layer-5
0layer_with_weights-3
0layer-6
1layer-7
2layer-8
3layer-9
4layer_with_weights-4
4layer-10
5layer-11
6layer_with_weights-5
6layer-12
7layer-13
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
ж
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

kernel
bias*
Б
D
_variables
E_iterations
F_learning_rate
G_index_dict
H
_momentums
I_velocities
J_update_step_xla*
)
Ktrace_0
Ltrace_1
Mtrace_2* 

Nserving_default* 
[U
VARIABLE_VALUEbatch_normalization_8/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_8/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!batch_normalization_8/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%batch_normalization_8/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_24/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_24/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_25/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_25/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_26/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_26/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_24/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_24/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_25/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_25/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_26/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_26/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
	1*

O0
P1*
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
О
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
╒
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]axis
	gamma
beta
moving_mean
moving_variance*
╚
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

kernel
bias
 d_jit_compiled_convolution_op*
О
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses* 
╚
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

kernel
bias
 q_jit_compiled_convolution_op*
О
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses* 
╚
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

kernel
bias
 ~_jit_compiled_convolution_op*
У
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses* 
м
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses
Л_random_generator* 
Ф
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses* 
м
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses

kernel
bias*
м
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Ю_random_generator* 
м
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses

kernel
bias*
м
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses
л_random_generator* 
j
0
1
2
3
4
5
6
7
8
9
10
11
12
13*
Z
0
1
2
3
4
5
6
7
8
9
10
11*

м0
н1
о2* 
Ш
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
:
┤trace_0
╡trace_1
╢trace_2
╖trace_3* 
:
╕trace_0
╣trace_1
║trace_2
╗trace_3* 

0
1*

0
1*
* 
Ш
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

┴trace_0* 

┬trace_0* 
■
E0
├1
─2
┼3
╞4
╟5
╚6
╔7
╩8
╦9
╠10
═11
╬12
╧13
╨14
╤15
╥16
╙17
╘18
╒19
╓20
╫21
╪22
┘23
┌24
█25
▄26
▌27
▐28*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
├0
┼1
╟2
╔3
╦4
═5
╧6
╤7
╙8
╒9
╫10
┘11
█12
▌13*
x
─0
╞1
╚2
╩3
╠4
╬5
╨6
╥7
╘8
╓9
╪10
┌11
▄12
▐13*
* 
* 
* 
* 
* 
<
▀	variables
р	keras_api

сtotal

тcount*
M
у	variables
ф	keras_api

хtotal

цcount
ч
_fn_kwargs*
* 
* 
* 
Ц
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

эtrace_0
юtrace_1* 

яtrace_0
Ёtrace_1* 
 
0
1
2
3*

0
1*
* 
Ш
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

Ўtrace_0
ўtrace_1* 

°trace_0
∙trace_1* 
* 

0
1*

0
1*


м0* 
Ш
·non_trainable_variables
√layers
№metrics
 ¤layer_regularization_losses
■layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

 trace_0* 

Аtrace_0* 
* 
* 
* 
* 
Ц
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

Жtrace_0* 

Зtrace_0* 

0
1*

0
1*
* 
Ш
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

Нtrace_0* 

Оtrace_0* 
* 
* 
* 
* 
Ц
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 

Фtrace_0* 

Хtrace_0* 

0
1*

0
1*
* 
Ш
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

Ыtrace_0* 

Ьtrace_0* 
* 
* 
* 
* 
Ы
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses* 

вtrace_0* 

гtrace_0* 
* 
* 
* 
Ь
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses* 

йtrace_0
кtrace_1* 

лtrace_0
мtrace_1* 
* 
* 
* 
* 
Ь
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses* 

▓trace_0* 

│trace_0* 

0
1*

0
1*


н0* 
Ю
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses*

╣trace_0* 

║trace_0* 
* 
* 
* 
Ь
╗non_trainable_variables
╝layers
╜metrics
 ╛layer_regularization_losses
┐layer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses* 

└trace_0
┴trace_1* 

┬trace_0
├trace_1* 
* 

0
1*

0
1*


о0* 
Ю
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses*

╔trace_0* 

╩trace_0* 
* 
* 
* 
Ь
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses* 

╨trace_0
╤trace_1* 

╥trace_0
╙trace_1* 
* 

╘trace_0* 

╒trace_0* 

╓trace_0* 

0
1*
j
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713*
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
mg
VARIABLE_VALUE"Adam/m/batch_normalization_8/gamma1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/batch_normalization_8/gamma1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/m/batch_normalization_8/beta1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/v/batch_normalization_8/beta1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_24/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_24/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_24/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_24/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_25/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_25/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_25/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_25/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_26/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_26/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_26/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_26/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_24/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_24/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_24/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_24/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_25/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_25/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_25/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_25/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_26/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_26/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_26/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_26/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*

с0
т1*

▀	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

х0
ц1*

у	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
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
0
1*
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


м0* 
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


н0* 
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


о0* 
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
√
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/batch_normalization_8/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_8/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_8/beta/Read/ReadVariableOp5Adam/v/batch_normalization_8/beta/Read/ReadVariableOp+Adam/m/conv2d_24/kernel/Read/ReadVariableOp+Adam/v/conv2d_24/kernel/Read/ReadVariableOp)Adam/m/conv2d_24/bias/Read/ReadVariableOp)Adam/v/conv2d_24/bias/Read/ReadVariableOp+Adam/m/conv2d_25/kernel/Read/ReadVariableOp+Adam/v/conv2d_25/kernel/Read/ReadVariableOp)Adam/m/conv2d_25/bias/Read/ReadVariableOp)Adam/v/conv2d_25/bias/Read/ReadVariableOp+Adam/m/conv2d_26/kernel/Read/ReadVariableOp+Adam/v/conv2d_26/kernel/Read/ReadVariableOp)Adam/m/conv2d_26/bias/Read/ReadVariableOp)Adam/v/conv2d_26/bias/Read/ReadVariableOp*Adam/m/dense_24/kernel/Read/ReadVariableOp*Adam/v/dense_24/kernel/Read/ReadVariableOp(Adam/m/dense_24/bias/Read/ReadVariableOp(Adam/v/dense_24/bias/Read/ReadVariableOp*Adam/m/dense_25/kernel/Read/ReadVariableOp*Adam/v/dense_25/kernel/Read/ReadVariableOp(Adam/m/dense_25/bias/Read/ReadVariableOp(Adam/v/dense_25/bias/Read/ReadVariableOp*Adam/m/dense_26/kernel/Read/ReadVariableOp*Adam/v/dense_26/kernel/Read/ReadVariableOp(Adam/m/dense_26/bias/Read/ReadVariableOp(Adam/v/dense_26/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*?
Tin8
624	*
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
GPU 2J 8В *)
f$R"
 __inference__traced_save_2181014
О
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/bias	iterationlearning_rate"Adam/m/batch_normalization_8/gamma"Adam/v/batch_normalization_8/gamma!Adam/m/batch_normalization_8/beta!Adam/v/batch_normalization_8/betaAdam/m/conv2d_24/kernelAdam/v/conv2d_24/kernelAdam/m/conv2d_24/biasAdam/v/conv2d_24/biasAdam/m/conv2d_25/kernelAdam/v/conv2d_25/kernelAdam/m/conv2d_25/biasAdam/v/conv2d_25/biasAdam/m/conv2d_26/kernelAdam/v/conv2d_26/kernelAdam/m/conv2d_26/biasAdam/v/conv2d_26/biasAdam/m/dense_24/kernelAdam/v/dense_24/kernelAdam/m/dense_24/biasAdam/v/dense_24/biasAdam/m/dense_25/kernelAdam/v/dense_25/kernelAdam/m/dense_25/biasAdam/v/dense_25/biasAdam/m/dense_26/kernelAdam/v/dense_26/kernelAdam/m/dense_26/biasAdam/v/dense_26/biastotal_1count_1totalcount*>
Tin7
523*
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
GPU 2J 8В *,
f'R%
#__inference__traced_restore_2181174щ┬
з
H
,__inference_dropout_26_layer_call_fn_2180792

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_26_layer_call_and_return_conditional_losses_2179056a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ф

f
G__inference_dropout_25_layer_call_and_return_conditional_losses_2179155

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Х
i
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2178888

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2180580

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╘g
Ї
__inference_call_2113054

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: АE
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	АQ
5sequential_8_conv2d_26_conv2d_readvariableop_resource:ААE
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	АI
4sequential_8_dense_24_matmul_readvariableop_resource:АвАD
5sequential_8_dense_24_biasadd_readvariableop_resource:	АH
4sequential_8_dense_25_matmul_readvariableop_resource:
ААD
5sequential_8_dense_25_biasadd_readvariableop_resource:	А:
'dense_26_matmul_readvariableop_resource:	А6
(dense_26_biasadd_readvariableop_resource:
identityИвdense_26/BiasAdd/ReadVariableOpвdense_26/MatMul/ReadVariableOpвBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpвDsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в1sequential_8/batch_normalization_8/ReadVariableOpв3sequential_8/batch_normalization_8/ReadVariableOp_1в-sequential_8/conv2d_24/BiasAdd/ReadVariableOpв,sequential_8/conv2d_24/Conv2D/ReadVariableOpв-sequential_8/conv2d_25/BiasAdd/ReadVariableOpв,sequential_8/conv2d_25/Conv2D/ReadVariableOpв-sequential_8/conv2d_26/BiasAdd/ReadVariableOpв,sequential_8/conv2d_26/Conv2D/ReadVariableOpв,sequential_8/dense_24/BiasAdd/ReadVariableOpв+sequential_8/dense_24/MatMul/ReadVariableOpв,sequential_8/dense_25/BiasAdd/ReadVariableOpв+sequential_8/dense_25/MatMul/ReadVariableOpВ
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( к
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0°
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
а
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Ж
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╚
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
л
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ё
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
б
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЗ
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╔
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
м
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
б
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЗ
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Ч
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:         		Аm
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  п
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:         Авг
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╖
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0*(
_output_shapes
:         Ав
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╣
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0*(
_output_shapes
:         АЗ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ю
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:         i
IdentityIdentitydense_26/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╓
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2И
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2М
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▐
e
G__inference_dropout_26_layer_call_and_return_conditional_losses_2179056

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╗
о
%__inference_CNN_layer_call_fn_2180018

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallО
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
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_2179719o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┌
a
E__inference_lambda_8_layer_call_and_return_conditional_losses_2179243

inputs
identityl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ь
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskf
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Х
i
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2178900

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
У	
╥
7__inference_batch_normalization_8_layer_call_fn_2180531

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2178825Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ф

f
G__inference_dropout_25_layer_call_and_return_conditional_losses_2180763

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ВQ
Р	
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179519
lambda_8_input+
batch_normalization_8_2179465:+
batch_normalization_8_2179467:+
batch_normalization_8_2179469:+
batch_normalization_8_2179471:+
conv2d_24_2179474: 
conv2d_24_2179476: ,
conv2d_25_2179480: А 
conv2d_25_2179482:	А-
conv2d_26_2179486:АА 
conv2d_26_2179488:	А%
dense_24_2179494:АвА
dense_24_2179496:	А$
dense_25_2179500:
АА
dense_25_2179502:	А
identityИв-batch_normalization_8/StatefulPartitionedCallв!conv2d_24/StatefulPartitionedCallв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_25/StatefulPartitionedCallв!conv2d_26/StatefulPartitionedCallв dense_24/StatefulPartitionedCallв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв dense_25/StatefulPartitionedCallв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpв"dropout_24/StatefulPartitionedCallв"dropout_25/StatefulPartitionedCallв"dropout_26/StatefulPartitionedCall╔
lambda_8/PartitionedCallPartitionedCalllambda_8_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lambda_8_layer_call_and_return_conditional_losses_2179243К
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0batch_normalization_8_2179465batch_normalization_8_2179467batch_normalization_8_2179469batch_normalization_8_2179471*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2178856п
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_24_2179474conv2d_24_2179476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2178944ї
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         %% * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2178876г
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_2179480conv2d_25_2179482*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2178962Ў
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2178888г
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_2179486conv2d_26_2179488*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2178980Ў
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2178900∙
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_24_layer_call_and_return_conditional_losses_2179194т
flatten_8/PartitionedCallPartitionedCall+dropout_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2179000Р
 dense_24/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_24_2179494dense_24_2179496*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2179017Ц
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_25_layer_call_and_return_conditional_losses_2179155Щ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_25_2179500dense_25_2179502*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2179045Ц
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_26_layer_call_and_return_conditional_losses_2179122М
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_24_2179474*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_24_2179494*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Д
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_25_2179500* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: {
IdentityIdentity+dropout_26/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А┤
NoOpNoOp.^batch_normalization_8/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_8_input
е	
┤
__inference_loss_fn_2_2180841N
:dense_25_kernel_regularizer_l2loss_readvariableop_resource:
АА
identityИв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpо
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_25_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_25/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp
Щ
e
,__inference_dropout_24_layer_call_fn_2180684

inputs
identityИвStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_24_layer_call_and_return_conditional_losses_2179194x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         		А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
ЮL
Щ
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179071

inputs+
batch_normalization_8_2178920:+
batch_normalization_8_2178922:+
batch_normalization_8_2178924:+
batch_normalization_8_2178926:+
conv2d_24_2178945: 
conv2d_24_2178947: ,
conv2d_25_2178963: А 
conv2d_25_2178965:	А-
conv2d_26_2178981:АА 
conv2d_26_2178983:	А%
dense_24_2179018:АвА
dense_24_2179020:	А$
dense_25_2179046:
АА
dense_25_2179048:	А
identityИв-batch_normalization_8/StatefulPartitionedCallв!conv2d_24/StatefulPartitionedCallв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_25/StatefulPartitionedCallв!conv2d_26/StatefulPartitionedCallв dense_24/StatefulPartitionedCallв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв dense_25/StatefulPartitionedCallв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp┴
lambda_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lambda_8_layer_call_and_return_conditional_losses_2178918М
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0batch_normalization_8_2178920batch_normalization_8_2178922batch_normalization_8_2178924batch_normalization_8_2178926*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2178825п
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_24_2178945conv2d_24_2178947*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2178944ї
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         %% * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2178876г
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_2178963conv2d_25_2178965*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2178962Ў
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2178888г
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_2178981conv2d_26_2178983*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2178980Ў
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2178900щ
dropout_24/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_24_layer_call_and_return_conditional_losses_2178992┌
flatten_8/PartitionedCallPartitionedCall#dropout_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2179000Р
 dense_24/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_24_2179018dense_24_2179020*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2179017с
dropout_25/PartitionedCallPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_25_layer_call_and_return_conditional_losses_2179028С
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_25_2179046dense_25_2179048*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2179045с
dropout_26/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_26_layer_call_and_return_conditional_losses_2179056М
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_24_2178945*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_24_2179018*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Д
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_25_2179046* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity#dropout_26/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А┼
NoOpNoOp.^batch_normalization_8/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▐%
╝
@__inference_CNN_layer_call_and_return_conditional_losses_2179891
input_1"
sequential_8_2179844:"
sequential_8_2179846:"
sequential_8_2179848:"
sequential_8_2179850:.
sequential_8_2179852: "
sequential_8_2179854: /
sequential_8_2179856: А#
sequential_8_2179858:	А0
sequential_8_2179860:АА#
sequential_8_2179862:	А)
sequential_8_2179864:АвА#
sequential_8_2179866:	А(
sequential_8_2179868:
АА#
sequential_8_2179870:	А#
dense_26_2179873:	А
dense_26_2179875:
identityИв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpв dense_26/StatefulPartitionedCallв$sequential_8/StatefulPartitionedCallг
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_8_2179844sequential_8_2179846sequential_8_2179848sequential_8_2179850sequential_8_2179852sequential_8_2179854sequential_8_2179856sequential_8_2179858sequential_8_2179860sequential_8_2179862sequential_8_2179864sequential_8_2179866sequential_8_2179868sequential_8_2179870*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179339Ъ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0dense_26_2179873dense_26_2179875*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2179566П
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179852*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179864*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179868* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_26/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
╠
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_2180712

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         АвZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         Ав"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
█%
╗
@__inference_CNN_layer_call_and_return_conditional_losses_2179719

inputs"
sequential_8_2179672:"
sequential_8_2179674:"
sequential_8_2179676:"
sequential_8_2179678:.
sequential_8_2179680: "
sequential_8_2179682: /
sequential_8_2179684: А#
sequential_8_2179686:	А0
sequential_8_2179688:АА#
sequential_8_2179690:	А)
sequential_8_2179692:АвА#
sequential_8_2179694:	А(
sequential_8_2179696:
АА#
sequential_8_2179698:	А#
dense_26_2179701:	А
dense_26_2179703:
identityИв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpв dense_26/StatefulPartitionedCallв$sequential_8/StatefulPartitionedCallв
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinputssequential_8_2179672sequential_8_2179674sequential_8_2179676sequential_8_2179678sequential_8_2179680sequential_8_2179682sequential_8_2179684sequential_8_2179686sequential_8_2179688sequential_8_2179690sequential_8_2179692sequential_8_2179694sequential_8_2179696sequential_8_2179698*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179339Ъ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0dense_26_2179701dense_26_2179703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2179566П
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179680*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179692*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179696* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_26/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Т
┤
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2178944

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK Щ
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         KK м
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┌
a
E__inference_lambda_8_layer_call_and_return_conditional_losses_2178918

inputs
identityl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ь
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskf
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╛
п
%__inference_CNN_layer_call_fn_2179791
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_2179719o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
═
Э
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2178825

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
▐
e
G__inference_dropout_26_layer_call_and_return_conditional_losses_2180802

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
в
п
%__inference_signature_wrapper_2179944
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_2178803o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
Ф

f
G__inference_dropout_26_layer_call_and_return_conditional_losses_2180814

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╒_
└
 __inference__traced_save_2181014
file_prefix:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_batch_normalization_8_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_8_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_8_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_8_beta_read_readvariableop6
2savev2_adam_m_conv2d_24_kernel_read_readvariableop6
2savev2_adam_v_conv2d_24_kernel_read_readvariableop4
0savev2_adam_m_conv2d_24_bias_read_readvariableop4
0savev2_adam_v_conv2d_24_bias_read_readvariableop6
2savev2_adam_m_conv2d_25_kernel_read_readvariableop6
2savev2_adam_v_conv2d_25_kernel_read_readvariableop4
0savev2_adam_m_conv2d_25_bias_read_readvariableop4
0savev2_adam_v_conv2d_25_bias_read_readvariableop6
2savev2_adam_m_conv2d_26_kernel_read_readvariableop6
2savev2_adam_v_conv2d_26_kernel_read_readvariableop4
0savev2_adam_m_conv2d_26_bias_read_readvariableop4
0savev2_adam_v_conv2d_26_bias_read_readvariableop5
1savev2_adam_m_dense_24_kernel_read_readvariableop5
1savev2_adam_v_dense_24_kernel_read_readvariableop3
/savev2_adam_m_dense_24_bias_read_readvariableop3
/savev2_adam_v_dense_24_bias_read_readvariableop5
1savev2_adam_m_dense_25_kernel_read_readvariableop5
1savev2_adam_v_dense_25_kernel_read_readvariableop3
/savev2_adam_m_dense_25_bias_read_readvariableop3
/savev2_adam_v_dense_25_bias_read_readvariableop5
1savev2_adam_m_dense_26_kernel_read_readvariableop5
1savev2_adam_v_dense_26_kernel_read_readvariableop3
/savev2_adam_m_dense_26_bias_read_readvariableop3
/savev2_adam_v_dense_26_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ў
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*Я
valueХBТ3B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╙
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Х
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_batch_normalization_8_gamma_read_readvariableop=savev2_adam_v_batch_normalization_8_gamma_read_readvariableop<savev2_adam_m_batch_normalization_8_beta_read_readvariableop<savev2_adam_v_batch_normalization_8_beta_read_readvariableop2savev2_adam_m_conv2d_24_kernel_read_readvariableop2savev2_adam_v_conv2d_24_kernel_read_readvariableop0savev2_adam_m_conv2d_24_bias_read_readvariableop0savev2_adam_v_conv2d_24_bias_read_readvariableop2savev2_adam_m_conv2d_25_kernel_read_readvariableop2savev2_adam_v_conv2d_25_kernel_read_readvariableop0savev2_adam_m_conv2d_25_bias_read_readvariableop0savev2_adam_v_conv2d_25_bias_read_readvariableop2savev2_adam_m_conv2d_26_kernel_read_readvariableop2savev2_adam_v_conv2d_26_kernel_read_readvariableop0savev2_adam_m_conv2d_26_bias_read_readvariableop0savev2_adam_v_conv2d_26_bias_read_readvariableop1savev2_adam_m_dense_24_kernel_read_readvariableop1savev2_adam_v_dense_24_kernel_read_readvariableop/savev2_adam_m_dense_24_bias_read_readvariableop/savev2_adam_v_dense_24_bias_read_readvariableop1savev2_adam_m_dense_25_kernel_read_readvariableop1savev2_adam_v_dense_25_kernel_read_readvariableop/savev2_adam_m_dense_25_bias_read_readvariableop/savev2_adam_v_dense_25_bias_read_readvariableop1savev2_adam_m_dense_26_kernel_read_readvariableop1savev2_adam_v_dense_26_kernel_read_readvariableop/savev2_adam_m_dense_26_bias_read_readvariableop/savev2_adam_v_dense_26_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *A
dtypes7
523	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*ф
_input_shapes╥
╧: ::::: : : А:А:АА:А:АвА:А:
АА:А:	А:: : ::::: : : : : А: А:А:А:АА:АА:А:А:АвА:АвА:А:А:
АА:
АА:А:А:	А:	А::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: А:!

_output_shapes	
:А:.	*
(
_output_shapes
:АА:!


_output_shapes	
:А:'#
!
_output_shapes
:АвА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: А:-)
'
_output_shapes
: А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:. *
(
_output_shapes
:АА:!!

_output_shapes	
:А:!"

_output_shapes	
:А:'##
!
_output_shapes
:АвА:'$#
!
_output_shapes
:АвА:!%

_output_shapes	
:А:!&

_output_shapes	
:А:&'"
 
_output_shapes
:
АА:&("
 
_output_shapes
:
АА:!)

_output_shapes	
:А:!*

_output_shapes	
:А:%+!

_output_shapes
:	А:%,!

_output_shapes
:	А: -

_output_shapes
:: .

_output_shapes
::/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: 
╝
N
2__inference_max_pooling2d_24_layer_call_fn_2180609

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2178876Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
е

ў
E__inference_dense_26_layer_call_and_return_conditional_losses_2180492

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
С
В
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2178980

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╝╤
Ж 
#__inference__traced_restore_2181174
file_prefix:
,assignvariableop_batch_normalization_8_gamma:;
-assignvariableop_1_batch_normalization_8_beta:B
4assignvariableop_2_batch_normalization_8_moving_mean:F
8assignvariableop_3_batch_normalization_8_moving_variance:=
#assignvariableop_4_conv2d_24_kernel: /
!assignvariableop_5_conv2d_24_bias: >
#assignvariableop_6_conv2d_25_kernel: А0
!assignvariableop_7_conv2d_25_bias:	А?
#assignvariableop_8_conv2d_26_kernel:АА0
!assignvariableop_9_conv2d_26_bias:	А8
#assignvariableop_10_dense_24_kernel:АвА0
!assignvariableop_11_dense_24_bias:	А7
#assignvariableop_12_dense_25_kernel:
АА0
!assignvariableop_13_dense_25_bias:	А6
#assignvariableop_14_dense_26_kernel:	А/
!assignvariableop_15_dense_26_bias:'
assignvariableop_16_iteration:	 +
!assignvariableop_17_learning_rate: D
6assignvariableop_18_adam_m_batch_normalization_8_gamma:D
6assignvariableop_19_adam_v_batch_normalization_8_gamma:C
5assignvariableop_20_adam_m_batch_normalization_8_beta:C
5assignvariableop_21_adam_v_batch_normalization_8_beta:E
+assignvariableop_22_adam_m_conv2d_24_kernel: E
+assignvariableop_23_adam_v_conv2d_24_kernel: 7
)assignvariableop_24_adam_m_conv2d_24_bias: 7
)assignvariableop_25_adam_v_conv2d_24_bias: F
+assignvariableop_26_adam_m_conv2d_25_kernel: АF
+assignvariableop_27_adam_v_conv2d_25_kernel: А8
)assignvariableop_28_adam_m_conv2d_25_bias:	А8
)assignvariableop_29_adam_v_conv2d_25_bias:	АG
+assignvariableop_30_adam_m_conv2d_26_kernel:ААG
+assignvariableop_31_adam_v_conv2d_26_kernel:АА8
)assignvariableop_32_adam_m_conv2d_26_bias:	А8
)assignvariableop_33_adam_v_conv2d_26_bias:	А?
*assignvariableop_34_adam_m_dense_24_kernel:АвА?
*assignvariableop_35_adam_v_dense_24_kernel:АвА7
(assignvariableop_36_adam_m_dense_24_bias:	А7
(assignvariableop_37_adam_v_dense_24_bias:	А>
*assignvariableop_38_adam_m_dense_25_kernel:
АА>
*assignvariableop_39_adam_v_dense_25_kernel:
АА7
(assignvariableop_40_adam_m_dense_25_bias:	А7
(assignvariableop_41_adam_v_dense_25_bias:	А=
*assignvariableop_42_adam_m_dense_26_kernel:	А=
*assignvariableop_43_adam_v_dense_26_kernel:	А6
(assignvariableop_44_adam_m_dense_26_bias:6
(assignvariableop_45_adam_v_dense_26_bias:%
assignvariableop_46_total_1: %
assignvariableop_47_count_1: #
assignvariableop_48_total: #
assignvariableop_49_count: 
identity_51ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9∙
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*Я
valueХBТ3B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╓
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B а
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*т
_output_shapes╧
╠:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_8_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_8_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_8_moving_meanIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_8_moving_varianceIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_24_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_24_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_25_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_25_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_26_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_26_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_24_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_24_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_25_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_25_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_26_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_26_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_16AssignVariableOpassignvariableop_16_iterationIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_17AssignVariableOp!assignvariableop_17_learning_rateIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_m_batch_normalization_8_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_v_batch_normalization_8_gammaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_m_batch_normalization_8_betaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_v_batch_normalization_8_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_m_conv2d_24_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_v_conv2d_24_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_m_conv2d_24_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_v_conv2d_24_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_m_conv2d_25_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_v_conv2d_25_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_m_conv2d_25_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_v_conv2d_25_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_m_conv2d_26_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_v_conv2d_26_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_m_conv2d_26_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_v_conv2d_26_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_m_dense_24_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_v_dense_24_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_m_dense_24_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_v_dense_24_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_dense_25_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_dense_25_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_dense_25_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_dense_25_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_m_dense_26_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_v_dense_26_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_m_dense_26_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_v_dense_26_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_46AssignVariableOpassignvariableop_46_total_1Identity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_47AssignVariableOpassignvariableop_47_count_1Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_48AssignVariableOpassignvariableop_48_totalIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_49AssignVariableOpassignvariableop_49_countIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Ы	
Identity_50Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_51IdentityIdentity_50:output:0^NoOp_1*
T0*
_output_shapes
: И	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_51Identity_51:output:0*y
_input_shapesh
f: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
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
д
н
E__inference_dense_25_layer_call_and_return_conditional_losses_2179045

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         АТ
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
э`
Я
I__inference_sequential_8_layer_call_and_return_conditional_losses_2180374

inputs;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: C
(conv2d_25_conv2d_readvariableop_resource: А8
)conv2d_25_biasadd_readvariableop_resource:	АD
(conv2d_26_conv2d_readvariableop_resource:АА8
)conv2d_26_biasadd_readvariableop_resource:	А<
'dense_24_matmul_readvariableop_resource:АвА7
(dense_24_biasadd_readvariableop_resource:	А;
'dense_25_matmul_readvariableop_resource:
АА7
(dense_25_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_8/FusedBatchNormV3/ReadVariableOpв7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_8/ReadVariableOpв&batch_normalization_8/ReadVariableOp_1в conv2d_24/BiasAdd/ReadVariableOpвconv2d_24/Conv2D/ReadVariableOpв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв conv2d_25/BiasAdd/ReadVariableOpвconv2d_25/Conv2D/ReadVariableOpв conv2d_26/BiasAdd/ReadVariableOpвconv2d_26/Conv2D/ReadVariableOpвdense_24/BiasAdd/ReadVariableOpвdense_24/MatMul/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpвdense_25/BiasAdd/ReadVariableOpвdense_25/MatMul/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpu
lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Р
lambda_8/strided_sliceStridedSliceinputs%lambda_8/strided_slice/stack:output:0'lambda_8/strided_slice/stack_1:output:0'lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskО
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╜
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3lambda_8/strided_slice:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( Р
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╤
conv2d_24/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Ж
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK l
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:         KK о
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
С
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0╔
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
З
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аm
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ап
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
Т
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╔
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
З
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аm
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:         Ап
max_pooling2d_26/MaxPoolMaxPoolconv2d_26/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
}
dropout_24/IdentityIdentity!max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:         		А`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  И
flatten_8/ReshapeReshapedropout_24/Identity:output:0flatten_8/Const:output:0*
T0*)
_output_shapes
:         АвЙ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Р
dense_24/MatMulMatMulflatten_8/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*(
_output_shapes
:         Аo
dropout_25/IdentityIdentitydense_24/Relu:activations:0*
T0*(
_output_shapes
:         АИ
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
dense_25/MatMulMatMuldropout_25/Identity:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:         Аo
dropout_26/IdentityIdentitydense_25/Relu:activations:0*
T0*(
_output_shapes
:         Аг
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ь
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: l
IdentityIdentitydropout_26/Identity:output:0^NoOp*
T0*(
_output_shapes
:         А·
NoOpNoOp6^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ў
■
.__inference_sequential_8_layer_call_fn_2180297

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179339p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
└
п
%__inference_CNN_layer_call_fn_2179620
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_2179585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
ъP
И	
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179339

inputs+
batch_normalization_8_2179285:+
batch_normalization_8_2179287:+
batch_normalization_8_2179289:+
batch_normalization_8_2179291:+
conv2d_24_2179294: 
conv2d_24_2179296: ,
conv2d_25_2179300: А 
conv2d_25_2179302:	А-
conv2d_26_2179306:АА 
conv2d_26_2179308:	А%
dense_24_2179314:АвА
dense_24_2179316:	А$
dense_25_2179320:
АА
dense_25_2179322:	А
identityИв-batch_normalization_8/StatefulPartitionedCallв!conv2d_24/StatefulPartitionedCallв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_25/StatefulPartitionedCallв!conv2d_26/StatefulPartitionedCallв dense_24/StatefulPartitionedCallв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв dense_25/StatefulPartitionedCallв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpв"dropout_24/StatefulPartitionedCallв"dropout_25/StatefulPartitionedCallв"dropout_26/StatefulPartitionedCall┴
lambda_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lambda_8_layer_call_and_return_conditional_losses_2179243К
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0batch_normalization_8_2179285batch_normalization_8_2179287batch_normalization_8_2179289batch_normalization_8_2179291*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2178856п
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_24_2179294conv2d_24_2179296*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2178944ї
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         %% * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2178876г
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_2179300conv2d_25_2179302*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2178962Ў
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2178888г
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_2179306conv2d_26_2179308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2178980Ў
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2178900∙
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_24_layer_call_and_return_conditional_losses_2179194т
flatten_8/PartitionedCallPartitionedCall+dropout_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2179000Р
 dense_24/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_24_2179314dense_24_2179316*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2179017Ц
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_25_layer_call_and_return_conditional_losses_2179155Щ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_25_2179320dense_25_2179322*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2179045Ц
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_26_layer_call_and_return_conditional_losses_2179122М
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_24_2179294*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_24_2179314*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Д
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_25_2179320* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: {
IdentityIdentity+dropout_26/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А┤
NoOpNoOp.^batch_normalization_8/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┐
F
*__inference_lambda_8_layer_call_fn_2180497

inputs
identity╕
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lambda_8_layer_call_and_return_conditional_losses_2178918h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╝
N
2__inference_max_pooling2d_25_layer_call_fn_2180639

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2178888Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
┤
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2180604

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK Щ
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         KK м
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┐
F
*__inference_lambda_8_layer_call_fn_2180502

inputs
identity╕
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lambda_8_layer_call_and_return_conditional_losses_2179243h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
сe
Ї
__inference_call_2114263

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: АE
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	АQ
5sequential_8_conv2d_26_conv2d_readvariableop_resource:ААE
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	АI
4sequential_8_dense_24_matmul_readvariableop_resource:АвАD
5sequential_8_dense_24_biasadd_readvariableop_resource:	АH
4sequential_8_dense_25_matmul_readvariableop_resource:
ААD
5sequential_8_dense_25_biasadd_readvariableop_resource:	А:
'dense_26_matmul_readvariableop_resource:	А6
(dense_26_biasadd_readvariableop_resource:
identityИвdense_26/BiasAdd/ReadVariableOpвdense_26/MatMul/ReadVariableOpвBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpвDsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в1sequential_8/batch_normalization_8/ReadVariableOpв3sequential_8/batch_normalization_8/ReadVariableOp_1в-sequential_8/conv2d_24/BiasAdd/ReadVariableOpв,sequential_8/conv2d_24/Conv2D/ReadVariableOpв-sequential_8/conv2d_25/BiasAdd/ReadVariableOpв,sequential_8/conv2d_25/Conv2D/ReadVariableOpв-sequential_8/conv2d_26/BiasAdd/ReadVariableOpв,sequential_8/conv2d_26/Conv2D/ReadVariableOpв,sequential_8/dense_24/BiasAdd/ReadVariableOpв+sequential_8/dense_24/MatMul/ReadVariableOpв,sequential_8/dense_25/BiasAdd/ReadVariableOpв+sequential_8/dense_25/MatMul/ReadVariableOpВ
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ╝
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_maskи
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Г
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( к
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ё
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
а
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0║
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK ~
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*'
_output_shapes
:АKK └
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
л
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0ш
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
б
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А┴
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
м
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ш
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
б
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*(
_output_shapes
:АА┴
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
П
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*(
_output_shapes
:А		Аm
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  з
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*!
_output_shapes
:ААвг
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0п
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЯ
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▒
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААu
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0* 
_output_shapes
:
ААБ
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0* 
_output_shapes
:
ААв
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0▒
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЯ
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▒
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААu
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0* 
_output_shapes
:
ААБ
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0* 
_output_shapes
:
ААЗ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	АД
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Й
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А`
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*
_output_shapes
:	Аa
IdentityIdentitydense_26/Softmax:softmax:0^NoOp*
T0*
_output_shapes
:	А╓
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2И
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2М
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
╥Я
г
@__inference_CNN_layer_call_and_return_conditional_losses_2180207

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: АE
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	АQ
5sequential_8_conv2d_26_conv2d_readvariableop_resource:ААE
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	АI
4sequential_8_dense_24_matmul_readvariableop_resource:АвАD
5sequential_8_dense_24_biasadd_readvariableop_resource:	АH
4sequential_8_dense_25_matmul_readvariableop_resource:
ААD
5sequential_8_dense_25_biasadd_readvariableop_resource:	А:
'dense_26_matmul_readvariableop_resource:	А6
(dense_26_biasadd_readvariableop_resource:
identityИв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpвdense_26/BiasAdd/ReadVariableOpвdense_26/MatMul/ReadVariableOpв1sequential_8/batch_normalization_8/AssignNewValueв3sequential_8/batch_normalization_8/AssignNewValue_1вBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpвDsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в1sequential_8/batch_normalization_8/ReadVariableOpв3sequential_8/batch_normalization_8/ReadVariableOp_1в-sequential_8/conv2d_24/BiasAdd/ReadVariableOpв,sequential_8/conv2d_24/Conv2D/ReadVariableOpв-sequential_8/conv2d_25/BiasAdd/ReadVariableOpв,sequential_8/conv2d_25/Conv2D/ReadVariableOpв-sequential_8/conv2d_26/BiasAdd/ReadVariableOpв,sequential_8/conv2d_26/Conv2D/ReadVariableOpв,sequential_8/dense_24/BiasAdd/ReadVariableOpв+sequential_8/dense_24/MatMul/ReadVariableOpв,sequential_8/dense_25/BiasAdd/ReadVariableOpв+sequential_8/dense_25/MatMul/ReadVariableOpВ
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Щ
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╥
1sequential_8/batch_normalization_8/AssignNewValueAssignVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource@sequential_8/batch_normalization_8/FusedBatchNormV3:batch_mean:0C^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(▄
3sequential_8/batch_normalization_8/AssignNewValue_1AssignVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resourceDsequential_8/batch_normalization_8/FusedBatchNormV3:batch_variance:0E^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(к
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0°
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
а
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Ж
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╚
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
л
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ё
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
б
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЗ
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╔
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
м
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
б
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЗ
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
j
%sequential_8/dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?┼
#sequential_8/dropout_24/dropout/MulMul.sequential_8/max_pooling2d_26/MaxPool:output:0.sequential_8/dropout_24/dropout/Const:output:0*
T0*0
_output_shapes
:         		АГ
%sequential_8/dropout_24/dropout/ShapeShape.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*
_output_shapes
:┼
<sequential_8/dropout_24/dropout/random_uniform/RandomUniformRandomUniform.sequential_8/dropout_24/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype0s
.sequential_8/dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ў
,sequential_8/dropout_24/dropout/GreaterEqualGreaterEqualEsequential_8/dropout_24/dropout/random_uniform/RandomUniform:output:07sequential_8/dropout_24/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		Аl
'sequential_8/dropout_24/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    №
(sequential_8/dropout_24/dropout/SelectV2SelectV20sequential_8/dropout_24/dropout/GreaterEqual:z:0'sequential_8/dropout_24/dropout/Mul:z:00sequential_8/dropout_24/dropout/Const_1:output:0*
T0*0
_output_shapes
:         		Аm
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  ╖
sequential_8/flatten_8/ReshapeReshape1sequential_8/dropout_24/dropout/SelectV2:output:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:         Авг
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╖
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:         Аj
%sequential_8/dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @╖
#sequential_8/dropout_25/dropout/MulMul(sequential_8/dense_24/Relu:activations:0.sequential_8/dropout_25/dropout/Const:output:0*
T0*(
_output_shapes
:         А}
%sequential_8/dropout_25/dropout/ShapeShape(sequential_8/dense_24/Relu:activations:0*
T0*
_output_shapes
:╜
<sequential_8/dropout_25/dropout/random_uniform/RandomUniformRandomUniform.sequential_8/dropout_25/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0s
.sequential_8/dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?я
,sequential_8/dropout_25/dropout/GreaterEqualGreaterEqualEsequential_8/dropout_25/dropout/random_uniform/RandomUniform:output:07sequential_8/dropout_25/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аl
'sequential_8/dropout_25/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ї
(sequential_8/dropout_25/dropout/SelectV2SelectV20sequential_8/dropout_25/dropout/GreaterEqual:z:0'sequential_8/dropout_25/dropout/Mul:z:00sequential_8/dropout_25/dropout/Const_1:output:0*
T0*(
_output_shapes
:         Ав
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0┴
sequential_8/dense_25/MatMulMatMul1sequential_8/dropout_25/dropout/SelectV2:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:         Аj
%sequential_8/dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @╖
#sequential_8/dropout_26/dropout/MulMul(sequential_8/dense_25/Relu:activations:0.sequential_8/dropout_26/dropout/Const:output:0*
T0*(
_output_shapes
:         А}
%sequential_8/dropout_26/dropout/ShapeShape(sequential_8/dense_25/Relu:activations:0*
T0*
_output_shapes
:╜
<sequential_8/dropout_26/dropout/random_uniform/RandomUniformRandomUniform.sequential_8/dropout_26/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0s
.sequential_8/dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?я
,sequential_8/dropout_26/dropout/GreaterEqualGreaterEqualEsequential_8/dropout_26/dropout/random_uniform/RandomUniform:output:07sequential_8/dropout_26/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аl
'sequential_8/dropout_26/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ї
(sequential_8/dropout_26/dropout/SelectV2SelectV20sequential_8/dropout_26/dropout/GreaterEqual:z:0'sequential_8/dropout_26/dropout/Mul:z:00sequential_8/dropout_26/dropout/Const_1:output:0*
T0*(
_output_shapes
:         АЗ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0ж
dense_26/MatMulMatMul1sequential_8/dropout_26/dropout/SelectV2:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:         ░
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: й
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: и
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_26/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ▌
NoOpNoOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp2^sequential_8/batch_normalization_8/AssignNewValue4^sequential_8/batch_normalization_8/AssignNewValue_1C^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2f
1sequential_8/batch_normalization_8/AssignNewValue1sequential_8/batch_normalization_8/AssignNewValue2j
3sequential_8/batch_normalization_8/AssignNewValue_13sequential_8/batch_normalization_8/AssignNewValue_12И
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2М
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╖
G
+__inference_flatten_8_layer_call_fn_2180706

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2179000b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         Ав"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
┌
a
E__inference_lambda_8_layer_call_and_return_conditional_losses_2180510

inputs
identityl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ь
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskf
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Х
i
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2178876

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╠
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_2179000

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         АвZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         Ав"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╟
H
,__inference_dropout_24_layer_call_fn_2180679

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_24_layer_call_and_return_conditional_losses_2178992i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         		А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
Х
i
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2180674

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
∙
e
,__inference_dropout_26_layer_call_fn_2180797

inputs
identityИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_26_layer_call_and_return_conditional_losses_2179122p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
О
Ж
.__inference_sequential_8_layer_call_fn_2179403
lambda_8_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCalllambda_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179339p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_8_input
Ф

f
G__inference_dropout_26_layer_call_and_return_conditional_losses_2179122

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╠

f
G__inference_dropout_24_layer_call_and_return_conditional_losses_2180701

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         		АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=п
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ь
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:         		Аj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:         		А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
Н
Б
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2180634

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         %%Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         %% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         %% 
 
_user_specified_nameinputs
З
┴
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2178856

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ц
я
I__inference_sequential_8_layer_call_and_return_conditional_losses_2180472

inputs;
-batch_normalization_8_readvariableop_resource:=
/batch_normalization_8_readvariableop_1_resource:L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_24_conv2d_readvariableop_resource: 7
)conv2d_24_biasadd_readvariableop_resource: C
(conv2d_25_conv2d_readvariableop_resource: А8
)conv2d_25_biasadd_readvariableop_resource:	АD
(conv2d_26_conv2d_readvariableop_resource:АА8
)conv2d_26_biasadd_readvariableop_resource:	А<
'dense_24_matmul_readvariableop_resource:АвА7
(dense_24_biasadd_readvariableop_resource:	А;
'dense_25_matmul_readvariableop_resource:
АА7
(dense_25_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_8/AssignNewValueв&batch_normalization_8/AssignNewValue_1в5batch_normalization_8/FusedBatchNormV3/ReadVariableOpв7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_8/ReadVariableOpв&batch_normalization_8/ReadVariableOp_1в conv2d_24/BiasAdd/ReadVariableOpвconv2d_24/Conv2D/ReadVariableOpв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв conv2d_25/BiasAdd/ReadVariableOpвconv2d_25/Conv2D/ReadVariableOpв conv2d_26/BiasAdd/ReadVariableOpвconv2d_26/Conv2D/ReadVariableOpвdense_24/BiasAdd/ReadVariableOpвdense_24/MatMul/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpвdense_25/BiasAdd/ReadVariableOpвdense_25/MatMul/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpu
lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Р
lambda_8/strided_sliceStridedSliceinputs%lambda_8/strided_slice/stack:output:0'lambda_8/strided_slice/stack_1:output:0'lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskО
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╦
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3lambda_8/strided_slice:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Р
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╤
conv2d_24/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Ж
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK l
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:         KK о
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
С
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0╔
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
З
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аm
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ап
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
Т
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╔
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
З
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аm
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:         Ап
max_pooling2d_26/MaxPoolMaxPoolconv2d_26/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
]
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?Ю
dropout_24/dropout/MulMul!max_pooling2d_26/MaxPool:output:0!dropout_24/dropout/Const:output:0*
T0*0
_output_shapes
:         		Аi
dropout_24/dropout/ShapeShape!max_pooling2d_26/MaxPool:output:0*
T0*
_output_shapes
:л
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype0f
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╨
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А_
dropout_24/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╚
dropout_24/dropout/SelectV2SelectV2#dropout_24/dropout/GreaterEqual:z:0dropout_24/dropout/Mul:z:0#dropout_24/dropout/Const_1:output:0*
T0*0
_output_shapes
:         		А`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  Р
flatten_8/ReshapeReshape$dropout_24/dropout/SelectV2:output:0flatten_8/Const:output:0*
T0*)
_output_shapes
:         АвЙ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Р
dense_24/MatMulMatMulflatten_8/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*(
_output_shapes
:         А]
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Р
dropout_25/dropout/MulMuldense_24/Relu:activations:0!dropout_25/dropout/Const:output:0*
T0*(
_output_shapes
:         Аc
dropout_25/dropout/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:г
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0f
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╚
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А_
dropout_25/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    └
dropout_25/dropout/SelectV2SelectV2#dropout_25/dropout/GreaterEqual:z:0dropout_25/dropout/Mul:z:0#dropout_25/dropout/Const_1:output:0*
T0*(
_output_shapes
:         АИ
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ъ
dense_25/MatMulMatMul$dropout_25/dropout/SelectV2:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:         А]
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Р
dropout_26/dropout/MulMuldense_25/Relu:activations:0!dropout_26/dropout/Const:output:0*
T0*(
_output_shapes
:         Аc
dropout_26/dropout/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:г
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0f
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╚
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А_
dropout_26/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    └
dropout_26/dropout/SelectV2SelectV2#dropout_26/dropout/GreaterEqual:z:0dropout_26/dropout/Mul:z:0#dropout_26/dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аг
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ь
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: t
IdentityIdentity$dropout_26/dropout/SelectV2:output:0^NoOp*
T0*(
_output_shapes
:         А╩
NoOpNoOp%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2180562

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┌
a
E__inference_lambda_8_layer_call_and_return_conditional_losses_2180518

inputs
identityl
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                n
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ь
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskf
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ю
а
+__inference_conv2d_24_layer_call_fn_2180589

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2178944w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         KK `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Р
Ж
.__inference_sequential_8_layer_call_fn_2179102
lambda_8_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCalllambda_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179071p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_8_input
╜
о
%__inference_CNN_layer_call_fn_2179981

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallР
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
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_CNN_layer_call_and_return_conditional_losses_2179585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╢L
б
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179461
lambda_8_input+
batch_normalization_8_2179407:+
batch_normalization_8_2179409:+
batch_normalization_8_2179411:+
batch_normalization_8_2179413:+
conv2d_24_2179416: 
conv2d_24_2179418: ,
conv2d_25_2179422: А 
conv2d_25_2179424:	А-
conv2d_26_2179428:АА 
conv2d_26_2179430:	А%
dense_24_2179436:АвА
dense_24_2179438:	А$
dense_25_2179442:
АА
dense_25_2179444:	А
identityИв-batch_normalization_8/StatefulPartitionedCallв!conv2d_24/StatefulPartitionedCallв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_25/StatefulPartitionedCallв!conv2d_26/StatefulPartitionedCallв dense_24/StatefulPartitionedCallв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв dense_25/StatefulPartitionedCallв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp╔
lambda_8/PartitionedCallPartitionedCalllambda_8_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lambda_8_layer_call_and_return_conditional_losses_2178918М
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall!lambda_8/PartitionedCall:output:0batch_normalization_8_2179407batch_normalization_8_2179409batch_normalization_8_2179411batch_normalization_8_2179413*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2178825п
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_24_2179416conv2d_24_2179418*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2178944ї
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         %% * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2178876г
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_2179422conv2d_25_2179424*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2178962Ў
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2178888г
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_2179428conv2d_26_2179430*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2178980Ў
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2178900щ
dropout_24/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_24_layer_call_and_return_conditional_losses_2178992┌
flatten_8/PartitionedCallPartitionedCall#dropout_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         Ав* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_2179000Р
 dense_24/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_24_2179436dense_24_2179438*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2179017с
dropout_25/PartitionedCallPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_25_layer_call_and_return_conditional_losses_2179028С
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_25_2179442dense_25_2179444*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2179045с
dropout_26/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_26_layer_call_and_return_conditional_losses_2179056М
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_24_2179416*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_24_2179436*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Д
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_25_2179442* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity#dropout_26/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А┼
NoOpNoOp.^batch_normalization_8/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_25/StatefulPartitionedCall2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_8_input
╘g
Ї
__inference_call_2114407

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: АE
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	АQ
5sequential_8_conv2d_26_conv2d_readvariableop_resource:ААE
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	АI
4sequential_8_dense_24_matmul_readvariableop_resource:АвАD
5sequential_8_dense_24_biasadd_readvariableop_resource:	АH
4sequential_8_dense_25_matmul_readvariableop_resource:
ААD
5sequential_8_dense_25_biasadd_readvariableop_resource:	А:
'dense_26_matmul_readvariableop_resource:	А6
(dense_26_biasadd_readvariableop_resource:
identityИвdense_26/BiasAdd/ReadVariableOpвdense_26/MatMul/ReadVariableOpвBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpвDsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в1sequential_8/batch_normalization_8/ReadVariableOpв3sequential_8/batch_normalization_8/ReadVariableOp_1в-sequential_8/conv2d_24/BiasAdd/ReadVariableOpв,sequential_8/conv2d_24/Conv2D/ReadVariableOpв-sequential_8/conv2d_25/BiasAdd/ReadVariableOpв,sequential_8/conv2d_25/Conv2D/ReadVariableOpв-sequential_8/conv2d_26/BiasAdd/ReadVariableOpв,sequential_8/conv2d_26/Conv2D/ReadVariableOpв,sequential_8/dense_24/BiasAdd/ReadVariableOpв+sequential_8/dense_24/MatMul/ReadVariableOpв,sequential_8/dense_25/BiasAdd/ReadVariableOpв+sequential_8/dense_25/MatMul/ReadVariableOpВ
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( к
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0°
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
а
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Ж
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╚
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
л
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ё
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
б
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЗ
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╔
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
м
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
б
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЗ
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Ч
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:         		Аm
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  п
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:         Авг
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╖
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0*(
_output_shapes
:         Ав
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╣
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0*(
_output_shapes
:         АЗ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ю
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:         i
IdentityIdentitydense_26/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╓
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2И
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2М
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
р%
╝
@__inference_CNN_layer_call_and_return_conditional_losses_2179841
input_1"
sequential_8_2179794:"
sequential_8_2179796:"
sequential_8_2179798:"
sequential_8_2179800:.
sequential_8_2179802: "
sequential_8_2179804: /
sequential_8_2179806: А#
sequential_8_2179808:	А0
sequential_8_2179810:АА#
sequential_8_2179812:	А)
sequential_8_2179814:АвА#
sequential_8_2179816:	А(
sequential_8_2179818:
АА#
sequential_8_2179820:	А#
dense_26_2179823:	А
dense_26_2179825:
identityИв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpв dense_26/StatefulPartitionedCallв$sequential_8/StatefulPartitionedCallе
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_8_2179794sequential_8_2179796sequential_8_2179798sequential_8_2179800sequential_8_2179802sequential_8_2179804sequential_8_2179806sequential_8_2179808sequential_8_2179810sequential_8_2179812sequential_8_2179814sequential_8_2179816sequential_8_2179818sequential_8_2179820*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179071Ъ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0dense_26_2179823dense_26_2179825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2179566П
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179802*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179814*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179818* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_26/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
∙
e
,__inference_dropout_25_layer_call_fn_2180746

inputs
identityИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_25_layer_call_and_return_conditional_losses_2179155p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╠

f
G__inference_dropout_24_layer_call_and_return_conditional_losses_2179194

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         		АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=п
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ь
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*0
_output_shapes
:         		Аj
IdentityIdentitydropout/SelectV2:output:0*
T0*0
_output_shapes
:         		А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
°
■
.__inference_sequential_8_layer_call_fn_2180264

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А
	unknown_9:АвА

unknown_10:	А

unknown_11:
АА

unknown_12:	А
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179071p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
д
н
E__inference_dense_25_layer_call_and_return_conditional_losses_2180787

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         АТ
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▐
e
G__inference_dropout_25_layer_call_and_return_conditional_losses_2179028

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┬z
╣
@__inference_CNN_layer_call_and_return_conditional_losses_2180102

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: АE
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	АQ
5sequential_8_conv2d_26_conv2d_readvariableop_resource:ААE
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	АI
4sequential_8_dense_24_matmul_readvariableop_resource:АвАD
5sequential_8_dense_24_biasadd_readvariableop_resource:	АH
4sequential_8_dense_25_matmul_readvariableop_resource:
ААD
5sequential_8_dense_25_biasadd_readvariableop_resource:	А:
'dense_26_matmul_readvariableop_resource:	А6
(dense_26_biasadd_readvariableop_resource:
identityИв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpвdense_26/BiasAdd/ReadVariableOpвdense_26/MatMul/ReadVariableOpвBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpвDsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в1sequential_8/batch_normalization_8/ReadVariableOpв3sequential_8/batch_normalization_8/ReadVariableOp_1в-sequential_8/conv2d_24/BiasAdd/ReadVariableOpв,sequential_8/conv2d_24/Conv2D/ReadVariableOpв-sequential_8/conv2d_25/BiasAdd/ReadVariableOpв,sequential_8/conv2d_25/Conv2D/ReadVariableOpв-sequential_8/conv2d_26/BiasAdd/ReadVariableOpв,sequential_8/conv2d_26/Conv2D/ReadVariableOpв,sequential_8/dense_24/BiasAdd/ReadVariableOpв+sequential_8/dense_24/MatMul/ReadVariableOpв,sequential_8/dense_25/BiasAdd/ReadVariableOpв+sequential_8/dense_25/MatMul/ReadVariableOpВ
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( к
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0°
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
а
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Ж
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╚
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
л
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ё
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
б
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЗ
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╔
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
м
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
б
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЗ
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Ч
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*0
_output_shapes
:         		Аm
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  п
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*)
_output_shapes
:         Авг
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╖
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0*(
_output_shapes
:         Ав
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╣
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0*(
_output_shapes
:         АЗ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ю
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*'
_output_shapes
:         ░
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: й
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: и
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_26/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         є
NoOpNoOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2И
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2М
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Х
i
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2180644

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
сe
Ї
__inference_call_2114335

inputsH
:sequential_8_batch_normalization_8_readvariableop_resource:J
<sequential_8_batch_normalization_8_readvariableop_1_resource:Y
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:[
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_8_conv2d_24_conv2d_readvariableop_resource: D
6sequential_8_conv2d_24_biasadd_readvariableop_resource: P
5sequential_8_conv2d_25_conv2d_readvariableop_resource: АE
6sequential_8_conv2d_25_biasadd_readvariableop_resource:	АQ
5sequential_8_conv2d_26_conv2d_readvariableop_resource:ААE
6sequential_8_conv2d_26_biasadd_readvariableop_resource:	АI
4sequential_8_dense_24_matmul_readvariableop_resource:АвАD
5sequential_8_dense_24_biasadd_readvariableop_resource:	АH
4sequential_8_dense_25_matmul_readvariableop_resource:
ААD
5sequential_8_dense_25_biasadd_readvariableop_resource:	А:
'dense_26_matmul_readvariableop_resource:	А6
(dense_26_biasadd_readvariableop_resource:
identityИвdense_26/BiasAdd/ReadVariableOpвdense_26/MatMul/ReadVariableOpвBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpвDsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в1sequential_8/batch_normalization_8/ReadVariableOpв3sequential_8/batch_normalization_8/ReadVariableOp_1в-sequential_8/conv2d_24/BiasAdd/ReadVariableOpв,sequential_8/conv2d_24/Conv2D/ReadVariableOpв-sequential_8/conv2d_25/BiasAdd/ReadVariableOpв,sequential_8/conv2d_25/Conv2D/ReadVariableOpв-sequential_8/conv2d_26/BiasAdd/ReadVariableOpв,sequential_8/conv2d_26/Conv2D/ReadVariableOpв,sequential_8/dense_24/BiasAdd/ReadVariableOpв+sequential_8/dense_24/MatMul/ReadVariableOpв,sequential_8/dense_25/BiasAdd/ReadVariableOpв+sequential_8/dense_25/MatMul/ReadVariableOpВ
)sequential_8/lambda_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_8/lambda_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ╝
#sequential_8/lambda_8/strided_sliceStridedSliceinputs2sequential_8/lambda_8/strided_slice/stack:output:04sequential_8/lambda_8/strided_slice/stack_1:output:04sequential_8/lambda_8/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_maskи
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Г
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3,sequential_8/lambda_8/strided_slice:output:09sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( к
,sequential_8/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ё
sequential_8/conv2d_24/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:04sequential_8/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
а
-sequential_8/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0║
sequential_8/conv2d_24/BiasAddBiasAdd&sequential_8/conv2d_24/Conv2D:output:05sequential_8/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK ~
sequential_8/conv2d_24/ReluRelu'sequential_8/conv2d_24/BiasAdd:output:0*
T0*'
_output_shapes
:АKK └
%sequential_8/max_pooling2d_24/MaxPoolMaxPool)sequential_8/conv2d_24/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
л
,sequential_8/conv2d_25/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_25_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0ш
sequential_8/conv2d_25/Conv2DConv2D.sequential_8/max_pooling2d_24/MaxPool:output:04sequential_8/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
б
-sequential_8/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
sequential_8/conv2d_25/BiasAddBiasAdd&sequential_8/conv2d_25/Conv2D:output:05sequential_8/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А
sequential_8/conv2d_25/ReluRelu'sequential_8/conv2d_25/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А┴
%sequential_8/max_pooling2d_25/MaxPoolMaxPool)sequential_8/conv2d_25/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
м
,sequential_8/conv2d_26/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_26_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ш
sequential_8/conv2d_26/Conv2DConv2D.sequential_8/max_pooling2d_25/MaxPool:output:04sequential_8/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
б
-sequential_8/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_26_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
sequential_8/conv2d_26/BiasAddBiasAdd&sequential_8/conv2d_26/Conv2D:output:05sequential_8/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА
sequential_8/conv2d_26/ReluRelu'sequential_8/conv2d_26/BiasAdd:output:0*
T0*(
_output_shapes
:АА┴
%sequential_8/max_pooling2d_26/MaxPoolMaxPool)sequential_8/conv2d_26/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
П
 sequential_8/dropout_24/IdentityIdentity.sequential_8/max_pooling2d_26/MaxPool:output:0*
T0*(
_output_shapes
:А		Аm
sequential_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  з
sequential_8/flatten_8/ReshapeReshape)sequential_8/dropout_24/Identity:output:0%sequential_8/flatten_8/Const:output:0*
T0*!
_output_shapes
:ААвг
+sequential_8/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_24_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0п
sequential_8/dense_24/MatMulMatMul'sequential_8/flatten_8/Reshape:output:03sequential_8/dense_24/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЯ
,sequential_8/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▒
sequential_8/dense_24/BiasAddBiasAdd&sequential_8/dense_24/MatMul:product:04sequential_8/dense_24/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААu
sequential_8/dense_24/ReluRelu&sequential_8/dense_24/BiasAdd:output:0*
T0* 
_output_shapes
:
ААБ
 sequential_8/dropout_25/IdentityIdentity(sequential_8/dense_24/Relu:activations:0*
T0* 
_output_shapes
:
ААв
+sequential_8/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0▒
sequential_8/dense_25/MatMulMatMul)sequential_8/dropout_25/Identity:output:03sequential_8/dense_25/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЯ
,sequential_8/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▒
sequential_8/dense_25/BiasAddBiasAdd&sequential_8/dense_25/MatMul:product:04sequential_8/dense_25/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААu
sequential_8/dense_25/ReluRelu&sequential_8/dense_25/BiasAdd:output:0*
T0* 
_output_shapes
:
ААБ
 sequential_8/dropout_26/IdentityIdentity(sequential_8/dense_25/Relu:activations:0*
T0* 
_output_shapes
:
ААЗ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
dense_26/MatMulMatMul)sequential_8/dropout_26/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	АД
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Й
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А`
dense_26/SoftmaxSoftmaxdense_26/BiasAdd:output:0*
T0*
_output_shapes
:	Аa
IdentityIdentitydense_26/Softmax:softmax:0^NoOp*
T0*
_output_shapes
:	А╓
NoOpNoOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOpC^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1.^sequential_8/conv2d_24/BiasAdd/ReadVariableOp-^sequential_8/conv2d_24/Conv2D/ReadVariableOp.^sequential_8/conv2d_25/BiasAdd/ReadVariableOp-^sequential_8/conv2d_25/Conv2D/ReadVariableOp.^sequential_8/conv2d_26/BiasAdd/ReadVariableOp-^sequential_8/conv2d_26/Conv2D/ReadVariableOp-^sequential_8/dense_24/BiasAdd/ReadVariableOp,^sequential_8/dense_24/MatMul/ReadVariableOp-^sequential_8/dense_25/BiasAdd/ReadVariableOp,^sequential_8/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2И
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2М
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12^
-sequential_8/conv2d_24/BiasAdd/ReadVariableOp-sequential_8/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_24/Conv2D/ReadVariableOp,sequential_8/conv2d_24/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_25/BiasAdd/ReadVariableOp-sequential_8/conv2d_25/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_25/Conv2D/ReadVariableOp,sequential_8/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_26/BiasAdd/ReadVariableOp-sequential_8/conv2d_26/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_26/Conv2D/ReadVariableOp,sequential_8/conv2d_26/Conv2D/ReadVariableOp2\
,sequential_8/dense_24/BiasAdd/ReadVariableOp,sequential_8/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_24/MatMul/ReadVariableOp+sequential_8/dense_24/MatMul/ReadVariableOp2\
,sequential_8/dense_25/BiasAdd/ReadVariableOp,sequential_8/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_25/MatMul/ReadVariableOp+sequential_8/dense_25/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
ї
г
+__inference_conv2d_26_layer_call_fn_2180653

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2178980x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
е

ў
E__inference_dense_26_layer_call_and_return_conditional_losses_2179566

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ч
═
"__inference__wrapped_model_2178803
input_1
cnn_2178769:
cnn_2178771:
cnn_2178773:
cnn_2178775:%
cnn_2178777: 
cnn_2178779: &
cnn_2178781: А
cnn_2178783:	А'
cnn_2178785:АА
cnn_2178787:	А 
cnn_2178789:АвА
cnn_2178791:	А
cnn_2178793:
АА
cnn_2178795:	А
cnn_2178797:	А
cnn_2178799:
identityИвCNN/StatefulPartitionedCallК
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn_2178769cnn_2178771cnn_2178773cnn_2178775cnn_2178777cnn_2178779cnn_2178781cnn_2178783cnn_2178785cnn_2178787cnn_2178789cnn_2178791cnn_2178793cnn_2178795cnn_2178797cnn_2178799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *!
fR
__inference_call_2113054s
IdentityIdentity$CNN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d
NoOpNoOp^CNN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2:
CNN/StatefulPartitionedCallCNN/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
з	
╡
__inference_loss_fn_1_2180832O
:dense_24_kernel_regularizer_l2loss_readvariableop_resource:АвА
identityИв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpп
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_24_kernel_regularizer_l2loss_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_24/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp
▐
e
G__inference_dropout_25_layer_call_and_return_conditional_losses_2180751

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
■
e
G__inference_dropout_24_layer_call_and_return_conditional_losses_2180689

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         		Аd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╟
Ш
*__inference_dense_26_layer_call_fn_2180481

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2179566o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
■
e
G__inference_dropout_24_layer_call_and_return_conditional_losses_2178992

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         		Аd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		А:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╝
N
2__inference_max_pooling2d_26_layer_call_fn_2180669

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2178900Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
С
В
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2180664

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╦
Ъ
*__inference_dense_25_layer_call_fn_2180772

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2179045p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╬
Ы
*__inference_dense_24_layer_call_fn_2180721

inputs
unknown:АвА
	unknown_0:	А
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2179017p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
С	
╥
7__inference_batch_normalization_8_layer_call_fn_2180544

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2178856Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
▌%
╗
@__inference_CNN_layer_call_and_return_conditional_losses_2179585

inputs"
sequential_8_2179526:"
sequential_8_2179528:"
sequential_8_2179530:"
sequential_8_2179532:.
sequential_8_2179534: "
sequential_8_2179536: /
sequential_8_2179538: А#
sequential_8_2179540:	А0
sequential_8_2179542:АА#
sequential_8_2179544:	А)
sequential_8_2179546:АвА#
sequential_8_2179548:	А(
sequential_8_2179550:
АА#
sequential_8_2179552:	А#
dense_26_2179567:	А
dense_26_2179569:
identityИв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpв dense_26/StatefulPartitionedCallв$sequential_8/StatefulPartitionedCallд
$sequential_8/StatefulPartitionedCallStatefulPartitionedCallinputssequential_8_2179526sequential_8_2179528sequential_8_2179530sequential_8_2179532sequential_8_2179534sequential_8_2179536sequential_8_2179538sequential_8_2179540sequential_8_2179542sequential_8_2179544sequential_8_2179546sequential_8_2179548sequential_8_2179550sequential_8_2179552*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179071Ъ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall-sequential_8/StatefulPartitionedCall:output:0dense_26_2179567dense_26_2179569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2179566П
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179534*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179546*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_8_2179550* 
_output_shapes
:
АА*
dtype0И
"dense_25/kernel/Regularizer/L2LossL2Loss9dense_25/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0+dense_25/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_25/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_26/StatefulPartitionedCall%^sequential_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp1dense_25/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2L
$sequential_8/StatefulPartitionedCall$sequential_8/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┐	
╝
__inference_loss_fn_0_2180823U
;conv2d_24_kernel_regularizer_l2loss_readvariableop_resource: 
identityИв2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp╢
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;conv2d_24_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_24/kernel/Regularizer/L2LossL2Loss:conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_24/kernel/Regularizer/mulMul+conv2d_24/kernel/Regularizer/mul/x:output:0,conv2d_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_24/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_24/kernel/Regularizer/L2Loss/ReadVariableOp
Х
i
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2180614

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
з
H
,__inference_dropout_25_layer_call_fn_2180741

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_25_layer_call_and_return_conditional_losses_2179028a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Є
в
+__inference_conv2d_25_layer_call_fn_2180623

inputs"
unknown: А
	unknown_0:	А
identityИвStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2178962x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         %%А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         %% : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         %% 
 
_user_specified_nameinputs
й
о
E__inference_dense_24_layer_call_and_return_conditional_losses_2180736

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         АУ
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
й
о
E__inference_dense_24_layer_call_and_return_conditional_losses_2179017

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         АУ
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_24/kernel/Regularizer/L2LossL2Loss9dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0+dense_24/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp1dense_24/kernel/Regularizer/L2Loss/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
Н
Б
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2178962

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         %%Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         %% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         %% 
 
_user_specified_nameinputs"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
C
input_18
serving_default_input_1:0         KK<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:╜┬
Д
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

h2ptjl
	_output

	optimizer
call

signatures"
_tf_keras_model
Ц
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
Ж
0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╜
"trace_0
#trace_1
$trace_2
%trace_32╥
%__inference_CNN_layer_call_fn_2179620
%__inference_CNN_layer_call_fn_2179981
%__inference_CNN_layer_call_fn_2180018
%__inference_CNN_layer_call_fn_2179791│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z"trace_0z#trace_1z$trace_2z%trace_3
й
&trace_0
'trace_1
(trace_2
)trace_32╛
@__inference_CNN_layer_call_and_return_conditional_losses_2180102
@__inference_CNN_layer_call_and_return_conditional_losses_2180207
@__inference_CNN_layer_call_and_return_conditional_losses_2179841
@__inference_CNN_layer_call_and_return_conditional_losses_2179891│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z&trace_0z'trace_1z(trace_2z)trace_3
═B╩
"__inference__wrapped_model_2178803input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
А
*layer-0
+layer_with_weights-0
+layer-1
,layer_with_weights-1
,layer-2
-layer-3
.layer_with_weights-2
.layer-4
/layer-5
0layer_with_weights-3
0layer-6
1layer-7
2layer-8
3layer-9
4layer_with_weights-4
4layer-10
5layer-11
6layer_with_weights-5
6layer-12
7layer-13
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_sequential
╗
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ь
D
_variables
E_iterations
F_learning_rate
G_index_dict
H
_momentums
I_velocities
J_update_step_xla"
experimentalOptimizer
╒
Ktrace_0
Ltrace_1
Mtrace_22Д
__inference_call_2114263
__inference_call_2114335
__inference_call_2114407│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zKtrace_0zLtrace_1zMtrace_2
,
Nserving_default"
signature_map
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
*:( 2conv2d_24/kernel
: 2conv2d_24/bias
+:) А2conv2d_25/kernel
:А2conv2d_25/bias
,:*АА2conv2d_26/kernel
:А2conv2d_26/bias
$:"АвА2dense_24/kernel
:А2dense_24/bias
#:!
АА2dense_25/kernel
:А2dense_25/bias
": 	А2dense_26/kernel
:2dense_26/bias
.
0
1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
%__inference_CNN_layer_call_fn_2179620input_1"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъBч
%__inference_CNN_layer_call_fn_2179981inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъBч
%__inference_CNN_layer_call_fn_2180018inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
%__inference_CNN_layer_call_fn_2179791input_1"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
@__inference_CNN_layer_call_and_return_conditional_losses_2180102inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
@__inference_CNN_layer_call_and_return_conditional_losses_2180207inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
@__inference_CNN_layer_call_and_return_conditional_losses_2179841input_1"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
@__inference_CNN_layer_call_and_return_conditional_losses_2179891input_1"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
е
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
ъ
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
▌
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

kernel
bias
 d_jit_compiled_convolution_op"
_tf_keras_layer
е
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

kernel
bias
 q_jit_compiled_convolution_op"
_tf_keras_layer
е
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

kernel
bias
 ~_jit_compiled_convolution_op"
_tf_keras_layer
к
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses
Л_random_generator"
_tf_keras_layer
л
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
┴
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
├
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Ю_random_generator"
_tf_keras_layer
┴
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
├
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses
л_random_generator"
_tf_keras_layer
Ж
0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
8
м0
н1
о2"
trackable_list_wrapper
▓
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ї
┤trace_0
╡trace_1
╢trace_2
╖trace_32В
.__inference_sequential_8_layer_call_fn_2179102
.__inference_sequential_8_layer_call_fn_2180264
.__inference_sequential_8_layer_call_fn_2180297
.__inference_sequential_8_layer_call_fn_2179403┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┤trace_0z╡trace_1z╢trace_2z╖trace_3
с
╕trace_0
╣trace_1
║trace_2
╗trace_32ю
I__inference_sequential_8_layer_call_and_return_conditional_losses_2180374
I__inference_sequential_8_layer_call_and_return_conditional_losses_2180472
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179461
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179519┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╕trace_0z╣trace_1z║trace_2z╗trace_3
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Ё
┴trace_02╤
*__inference_dense_26_layer_call_fn_2180481в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┴trace_0
Л
┬trace_02ь
E__inference_dense_26_layer_call_and_return_conditional_losses_2180492в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┬trace_0
Ъ
E0
├1
─2
┼3
╞4
╟5
╚6
╔7
╩8
╦9
╠10
═11
╬12
╧13
╨14
╤15
╥16
╙17
╘18
╒19
╓20
╫21
╪22
┘23
┌24
█25
▄26
▌27
▐28"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
Ф
├0
┼1
╟2
╔3
╦4
═5
╧6
╤7
╙8
╒9
╫10
┘11
█12
▌13"
trackable_list_wrapper
Ф
─0
╞1
╚2
╩3
╠4
╬5
╨6
╥7
╘8
╓9
╪10
┌11
▄12
▐13"
trackable_list_wrapper
┐2╝╣
о▓к
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
▌B┌
__inference_call_2114263inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▌B┌
__inference_call_2114335inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▌B┌
__inference_call_2114407inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠B╔
%__inference_signature_wrapper_2179944input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
▀	variables
р	keras_api

сtotal

тcount"
_tf_keras_metric
c
у	variables
ф	keras_api

хtotal

цcount
ч
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
╒
эtrace_0
юtrace_12Ъ
*__inference_lambda_8_layer_call_fn_2180497
*__inference_lambda_8_layer_call_fn_2180502┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zэtrace_0zюtrace_1
Л
яtrace_0
Ёtrace_12╨
E__inference_lambda_8_layer_call_and_return_conditional_losses_2180510
E__inference_lambda_8_layer_call_and_return_conditional_losses_2180518┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0zЁtrace_1
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ёnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
у
Ўtrace_0
ўtrace_12и
7__inference_batch_normalization_8_layer_call_fn_2180531
7__inference_batch_normalization_8_layer_call_fn_2180544│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЎtrace_0zўtrace_1
Щ
°trace_0
∙trace_12▐
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2180562
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2180580│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0z∙trace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
м0"
trackable_list_wrapper
▓
·non_trainable_variables
√layers
№metrics
 ¤layer_regularization_losses
■layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ё
 trace_02╥
+__inference_conv2d_24_layer_call_fn_2180589в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z trace_0
М
Аtrace_02э
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2180604в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zАtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
°
Жtrace_02┘
2__inference_max_pooling2d_24_layer_call_fn_2180609в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
У
Зtrace_02Ї
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2180614в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЗtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
ё
Нtrace_02╥
+__inference_conv2d_25_layer_call_fn_2180623в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
М
Оtrace_02э
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2180634в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zОtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
°
Фtrace_02┘
2__inference_max_pooling2d_25_layer_call_fn_2180639в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
У
Хtrace_02Ї
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2180644в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zХtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
ё
Ыtrace_02╥
+__inference_conv2d_26_layer_call_fn_2180653в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
М
Ьtrace_02э
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2180664в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЬtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╖
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
	variables
Аtrainable_variables
Бregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
°
вtrace_02┘
2__inference_max_pooling2d_26_layer_call_fn_2180669в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0
У
гtrace_02Ї
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2180674в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zгtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
═
йtrace_0
кtrace_12Т
,__inference_dropout_24_layer_call_fn_2180679
,__inference_dropout_24_layer_call_fn_2180684│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zйtrace_0zкtrace_1
Г
лtrace_0
мtrace_12╚
G__inference_dropout_24_layer_call_and_return_conditional_losses_2180689
G__inference_dropout_24_layer_call_and_return_conditional_losses_2180701│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zлtrace_0zмtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
нnon_trainable_variables
оlayers
пmetrics
 ░layer_regularization_losses
▒layer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
ё
▓trace_02╥
+__inference_flatten_8_layer_call_fn_2180706в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▓trace_0
М
│trace_02э
F__inference_flatten_8_layer_call_and_return_conditional_losses_2180712в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z│trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
н0"
trackable_list_wrapper
╕
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
Ё
╣trace_02╤
*__inference_dense_24_layer_call_fn_2180721в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╣trace_0
Л
║trace_02ь
E__inference_dense_24_layer_call_and_return_conditional_losses_2180736в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z║trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╗non_trainable_variables
╝layers
╜metrics
 ╛layer_regularization_losses
┐layer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
═
└trace_0
┴trace_12Т
,__inference_dropout_25_layer_call_fn_2180741
,__inference_dropout_25_layer_call_fn_2180746│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z└trace_0z┴trace_1
Г
┬trace_0
├trace_12╚
G__inference_dropout_25_layer_call_and_return_conditional_losses_2180751
G__inference_dropout_25_layer_call_and_return_conditional_losses_2180763│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┬trace_0z├trace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
о0"
trackable_list_wrapper
╕
─non_trainable_variables
┼layers
╞metrics
 ╟layer_regularization_losses
╚layer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
Ё
╔trace_02╤
*__inference_dense_25_layer_call_fn_2180772в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╔trace_0
Л
╩trace_02ь
E__inference_dense_25_layer_call_and_return_conditional_losses_2180787в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
═
╨trace_0
╤trace_12Т
,__inference_dropout_26_layer_call_fn_2180792
,__inference_dropout_26_layer_call_fn_2180797│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╨trace_0z╤trace_1
Г
╥trace_0
╙trace_12╚
G__inference_dropout_26_layer_call_and_return_conditional_losses_2180802
G__inference_dropout_26_layer_call_and_return_conditional_losses_2180814│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0z╙trace_1
"
_generic_user_object
╨
╘trace_02▒
__inference_loss_fn_0_2180823П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z╘trace_0
╨
╒trace_02▒
__inference_loss_fn_1_2180832П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z╒trace_0
╨
╓trace_02▒
__inference_loss_fn_2_2180841П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z╓trace_0
.
0
1"
trackable_list_wrapper
Ж
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЗBД
.__inference_sequential_8_layer_call_fn_2179102lambda_8_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
.__inference_sequential_8_layer_call_fn_2180264inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
.__inference_sequential_8_layer_call_fn_2180297inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЗBД
.__inference_sequential_8_layer_call_fn_2179403lambda_8_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
I__inference_sequential_8_layer_call_and_return_conditional_losses_2180374inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
I__inference_sequential_8_layer_call_and_return_conditional_losses_2180472inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
вBЯ
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179461lambda_8_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
вBЯ
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179519lambda_8_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▐B█
*__inference_dense_26_layer_call_fn_2180481inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_dense_26_layer_call_and_return_conditional_losses_2180492inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.:,2"Adam/m/batch_normalization_8/gamma
.:,2"Adam/v/batch_normalization_8/gamma
-:+2!Adam/m/batch_normalization_8/beta
-:+2!Adam/v/batch_normalization_8/beta
/:- 2Adam/m/conv2d_24/kernel
/:- 2Adam/v/conv2d_24/kernel
!: 2Adam/m/conv2d_24/bias
!: 2Adam/v/conv2d_24/bias
0:. А2Adam/m/conv2d_25/kernel
0:. А2Adam/v/conv2d_25/kernel
": А2Adam/m/conv2d_25/bias
": А2Adam/v/conv2d_25/bias
1:/АА2Adam/m/conv2d_26/kernel
1:/АА2Adam/v/conv2d_26/kernel
": А2Adam/m/conv2d_26/bias
": А2Adam/v/conv2d_26/bias
):'АвА2Adam/m/dense_24/kernel
):'АвА2Adam/v/dense_24/kernel
!:А2Adam/m/dense_24/bias
!:А2Adam/v/dense_24/bias
(:&
АА2Adam/m/dense_25/kernel
(:&
АА2Adam/v/dense_25/kernel
!:А2Adam/m/dense_25/bias
!:А2Adam/v/dense_25/bias
':%	А2Adam/m/dense_26/kernel
':%	А2Adam/v/dense_26/kernel
 :2Adam/m/dense_26/bias
 :2Adam/v/dense_26/bias
0
с0
т1"
trackable_list_wrapper
.
▀	variables"
_generic_user_object
:  (2total
:  (2count
0
х0
ц1"
trackable_list_wrapper
.
у	variables"
_generic_user_object
:  (2total
:  (2count
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
√B°
*__inference_lambda_8_layer_call_fn_2180497inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
*__inference_lambda_8_layer_call_fn_2180502inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
E__inference_lambda_8_layer_call_and_return_conditional_losses_2180510inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
E__inference_lambda_8_layer_call_and_return_conditional_losses_2180518inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
7__inference_batch_normalization_8_layer_call_fn_2180531inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
7__inference_batch_normalization_8_layer_call_fn_2180544inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2180562inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2180580inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
м0"
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_conv2d_24_layer_call_fn_2180589inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2180604inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
цBу
2__inference_max_pooling2d_24_layer_call_fn_2180609inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
БB■
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2180614inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
+__inference_conv2d_25_layer_call_fn_2180623inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2180634inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
цBу
2__inference_max_pooling2d_25_layer_call_fn_2180639inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
БB■
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2180644inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
+__inference_conv2d_26_layer_call_fn_2180653inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2180664inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
цBу
2__inference_max_pooling2d_26_layer_call_fn_2180669inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
БB■
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2180674inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
ёBю
,__inference_dropout_24_layer_call_fn_2180679inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
,__inference_dropout_24_layer_call_fn_2180684inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
G__inference_dropout_24_layer_call_and_return_conditional_losses_2180689inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
G__inference_dropout_24_layer_call_and_return_conditional_losses_2180701inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▀B▄
+__inference_flatten_8_layer_call_fn_2180706inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_flatten_8_layer_call_and_return_conditional_losses_2180712inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
н0"
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
*__inference_dense_24_layer_call_fn_2180721inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_dense_24_layer_call_and_return_conditional_losses_2180736inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
ёBю
,__inference_dropout_25_layer_call_fn_2180741inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
,__inference_dropout_25_layer_call_fn_2180746inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
G__inference_dropout_25_layer_call_and_return_conditional_losses_2180751inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
G__inference_dropout_25_layer_call_and_return_conditional_losses_2180763inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
о0"
trackable_list_wrapper
 "
trackable_dict_wrapper
▐B█
*__inference_dense_25_layer_call_fn_2180772inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
E__inference_dense_25_layer_call_and_return_conditional_losses_2180787inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
ёBю
,__inference_dropout_26_layer_call_fn_2180792inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
,__inference_dropout_26_layer_call_fn_2180797inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
G__inference_dropout_26_layer_call_and_return_conditional_losses_2180802inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
G__inference_dropout_26_layer_call_and_return_conditional_losses_2180814inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┤B▒
__inference_loss_fn_0_2180823"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
┤B▒
__inference_loss_fn_1_2180832"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
┤B▒
__inference_loss_fn_2_2180841"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в ┬
@__inference_CNN_layer_call_and_return_conditional_losses_2179841~<в9
2в/
)К&
input_1         KK
p 
к ",в)
"К
tensor_0         
Ъ ┬
@__inference_CNN_layer_call_and_return_conditional_losses_2179891~<в9
2в/
)К&
input_1         KK
p
к ",в)
"К
tensor_0         
Ъ ┴
@__inference_CNN_layer_call_and_return_conditional_losses_2180102};в8
1в.
(К%
inputs         KK
p 
к ",в)
"К
tensor_0         
Ъ ┴
@__inference_CNN_layer_call_and_return_conditional_losses_2180207};в8
1в.
(К%
inputs         KK
p
к ",в)
"К
tensor_0         
Ъ Ь
%__inference_CNN_layer_call_fn_2179620s<в9
2в/
)К&
input_1         KK
p 
к "!К
unknown         Ь
%__inference_CNN_layer_call_fn_2179791s<в9
2в/
)К&
input_1         KK
p
к "!К
unknown         Ы
%__inference_CNN_layer_call_fn_2179981r;в8
1в.
(К%
inputs         KK
p 
к "!К
unknown         Ы
%__inference_CNN_layer_call_fn_2180018r;в8
1в.
(К%
inputs         KK
p
к "!К
unknown         и
"__inference__wrapped_model_2178803Б8в5
.в+
)К&
input_1         KK
к "3к0
.
output_1"К
output_1         Ї
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2180562ЭMвJ
Cв@
:К7
inputs+                           
p 
к "FвC
<К9
tensor_0+                           
Ъ Ї
R__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2180580ЭMвJ
Cв@
:К7
inputs+                           
p
к "FвC
<К9
tensor_0+                           
Ъ ╬
7__inference_batch_normalization_8_layer_call_fn_2180531ТMвJ
Cв@
:К7
inputs+                           
p 
к ";К8
unknown+                           ╬
7__inference_batch_normalization_8_layer_call_fn_2180544ТMвJ
Cв@
:К7
inputs+                           
p
к ";К8
unknown+                           ~
__inference_call_2114263b3в0
)в&
 К
inputsАKK
p
к "К
unknown	А~
__inference_call_2114335b3в0
)в&
 К
inputsАKK
p 
к "К
unknown	АО
__inference_call_2114407r;в8
1в.
(К%
inputs         KK
p 
к "!К
unknown         ╜
F__inference_conv2d_24_layer_call_and_return_conditional_losses_2180604s7в4
-в*
(К%
inputs         KK
к "4в1
*К'
tensor_0         KK 
Ъ Ч
+__inference_conv2d_24_layer_call_fn_2180589h7в4
-в*
(К%
inputs         KK
к ")К&
unknown         KK ╛
F__inference_conv2d_25_layer_call_and_return_conditional_losses_2180634t7в4
-в*
(К%
inputs         %% 
к "5в2
+К(
tensor_0         %%А
Ъ Ш
+__inference_conv2d_25_layer_call_fn_2180623i7в4
-в*
(К%
inputs         %% 
к "*К'
unknown         %%А┐
F__inference_conv2d_26_layer_call_and_return_conditional_losses_2180664u8в5
.в+
)К&
inputs         А
к "5в2
+К(
tensor_0         А
Ъ Щ
+__inference_conv2d_26_layer_call_fn_2180653j8в5
.в+
)К&
inputs         А
к "*К'
unknown         Ап
E__inference_dense_24_layer_call_and_return_conditional_losses_2180736f1в.
'в$
"К
inputs         Ав
к "-в*
#К 
tensor_0         А
Ъ Й
*__inference_dense_24_layer_call_fn_2180721[1в.
'в$
"К
inputs         Ав
к ""К
unknown         Ао
E__inference_dense_25_layer_call_and_return_conditional_losses_2180787e0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ И
*__inference_dense_25_layer_call_fn_2180772Z0в-
&в#
!К
inputs         А
к ""К
unknown         Ан
E__inference_dense_26_layer_call_and_return_conditional_losses_2180492d0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ З
*__inference_dense_26_layer_call_fn_2180481Y0в-
&в#
!К
inputs         А
к "!К
unknown         └
G__inference_dropout_24_layer_call_and_return_conditional_losses_2180689u<в9
2в/
)К&
inputs         		А
p 
к "5в2
+К(
tensor_0         		А
Ъ └
G__inference_dropout_24_layer_call_and_return_conditional_losses_2180701u<в9
2в/
)К&
inputs         		А
p
к "5в2
+К(
tensor_0         		А
Ъ Ъ
,__inference_dropout_24_layer_call_fn_2180679j<в9
2в/
)К&
inputs         		А
p 
к "*К'
unknown         		АЪ
,__inference_dropout_24_layer_call_fn_2180684j<в9
2в/
)К&
inputs         		А
p
к "*К'
unknown         		А░
G__inference_dropout_25_layer_call_and_return_conditional_losses_2180751e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ ░
G__inference_dropout_25_layer_call_and_return_conditional_losses_2180763e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ К
,__inference_dropout_25_layer_call_fn_2180741Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         АК
,__inference_dropout_25_layer_call_fn_2180746Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         А░
G__inference_dropout_26_layer_call_and_return_conditional_losses_2180802e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ ░
G__inference_dropout_26_layer_call_and_return_conditional_losses_2180814e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ К
,__inference_dropout_26_layer_call_fn_2180792Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         АК
,__inference_dropout_26_layer_call_fn_2180797Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         А┤
F__inference_flatten_8_layer_call_and_return_conditional_losses_2180712j8в5
.в+
)К&
inputs         		А
к ".в+
$К!
tensor_0         Ав
Ъ О
+__inference_flatten_8_layer_call_fn_2180706_8в5
.в+
)К&
inputs         		А
к "#К 
unknown         Ав└
E__inference_lambda_8_layer_call_and_return_conditional_losses_2180510w?в<
5в2
(К%
inputs         KK

 
p 
к "4в1
*К'
tensor_0         KK
Ъ └
E__inference_lambda_8_layer_call_and_return_conditional_losses_2180518w?в<
5в2
(К%
inputs         KK

 
p
к "4в1
*К'
tensor_0         KK
Ъ Ъ
*__inference_lambda_8_layer_call_fn_2180497l?в<
5в2
(К%
inputs         KK

 
p 
к ")К&
unknown         KKЪ
*__inference_lambda_8_layer_call_fn_2180502l?в<
5в2
(К%
inputs         KK

 
p
к ")К&
unknown         KKE
__inference_loss_fn_0_2180823$в

в 
к "К
unknown E
__inference_loss_fn_1_2180832$в

в 
к "К
unknown E
__inference_loss_fn_2_2180841$в

в 
к "К
unknown ў
M__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_2180614еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_24_layer_call_fn_2180609ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ў
M__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_2180644еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_25_layer_call_fn_2180639ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ў
M__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2180674еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_26_layer_call_fn_2180669ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ╓
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179461ИGвD
=в:
0К-
lambda_8_input         KK
p 

 
к "-в*
#К 
tensor_0         А
Ъ ╓
I__inference_sequential_8_layer_call_and_return_conditional_losses_2179519ИGвD
=в:
0К-
lambda_8_input         KK
p

 
к "-в*
#К 
tensor_0         А
Ъ ╬
I__inference_sequential_8_layer_call_and_return_conditional_losses_2180374А?в<
5в2
(К%
inputs         KK
p 

 
к "-в*
#К 
tensor_0         А
Ъ ╬
I__inference_sequential_8_layer_call_and_return_conditional_losses_2180472А?в<
5в2
(К%
inputs         KK
p

 
к "-в*
#К 
tensor_0         А
Ъ п
.__inference_sequential_8_layer_call_fn_2179102}GвD
=в:
0К-
lambda_8_input         KK
p 

 
к ""К
unknown         Ап
.__inference_sequential_8_layer_call_fn_2179403}GвD
=в:
0К-
lambda_8_input         KK
p

 
к ""К
unknown         Аз
.__inference_sequential_8_layer_call_fn_2180264u?в<
5в2
(К%
inputs         KK
p 

 
к ""К
unknown         Аз
.__inference_sequential_8_layer_call_fn_2180297u?в<
5в2
(К%
inputs         KK
p

 
к ""К
unknown         А╢
%__inference_signature_wrapper_2179944МCв@
в 
9к6
4
input_1)К&
input_1         KK"3к0
.
output_1"К
output_1         