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
Adam/v/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_17/bias
y
(Adam/v/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/bias*
_output_shapes
:*
dtype0
А
Adam/m/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_17/bias
y
(Adam/m/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/bias*
_output_shapes
:*
dtype0
Й
Adam/v/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/v/dense_17/kernel
В
*Adam/v/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/kernel*
_output_shapes
:	А*
dtype0
Й
Adam/m/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/m/dense_17/kernel
В
*Adam/m/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/kernel*
_output_shapes
:	А*
dtype0
Б
Adam/v/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/dense_16/bias
z
(Adam/v/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/dense_16/bias
z
(Adam/m/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/bias*
_output_shapes	
:А*
dtype0
К
Adam/v/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/v/dense_16/kernel
Г
*Adam/v/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/kernel* 
_output_shapes
:
АА*
dtype0
К
Adam/m/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*'
shared_nameAdam/m/dense_16/kernel
Г
*Adam/m/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/kernel* 
_output_shapes
:
АА*
dtype0
Б
Adam/v/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/dense_15/bias
z
(Adam/v/dense_15/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_15/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/dense_15/bias
z
(Adam/m/dense_15/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_15/bias*
_output_shapes	
:А*
dtype0
Л
Adam/v/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*'
shared_nameAdam/v/dense_15/kernel
Д
*Adam/v/dense_15/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_15/kernel*!
_output_shapes
:АвА*
dtype0
Л
Adam/m/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*'
shared_nameAdam/m/dense_15/kernel
Д
*Adam/m/dense_15/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_15/kernel*!
_output_shapes
:АвА*
dtype0
Г
Adam/v/conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/v/conv2d_17/bias
|
)Adam/v/conv2d_17/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_17/bias*
_output_shapes	
:А*
dtype0
Г
Adam/m/conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/m/conv2d_17/bias
|
)Adam/m/conv2d_17/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_17/bias*
_output_shapes	
:А*
dtype0
Ф
Adam/v/conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/v/conv2d_17/kernel
Н
+Adam/v/conv2d_17/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_17/kernel*(
_output_shapes
:АА*
dtype0
Ф
Adam/m/conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/m/conv2d_17/kernel
Н
+Adam/m/conv2d_17/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_17/kernel*(
_output_shapes
:АА*
dtype0
Г
Adam/v/conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/v/conv2d_16/bias
|
)Adam/v/conv2d_16/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_16/bias*
_output_shapes	
:А*
dtype0
Г
Adam/m/conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/m/conv2d_16/bias
|
)Adam/m/conv2d_16/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_16/bias*
_output_shapes	
:А*
dtype0
У
Adam/v/conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/v/conv2d_16/kernel
М
+Adam/v/conv2d_16/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_16/kernel*'
_output_shapes
: А*
dtype0
У
Adam/m/conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/m/conv2d_16/kernel
М
+Adam/m/conv2d_16/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_16/kernel*'
_output_shapes
: А*
dtype0
В
Adam/v/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/conv2d_15/bias
{
)Adam/v/conv2d_15/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_15/bias*
_output_shapes
: *
dtype0
В
Adam/m/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/conv2d_15/bias
{
)Adam/m/conv2d_15/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_15/bias*
_output_shapes
: *
dtype0
Т
Adam/v/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/v/conv2d_15/kernel
Л
+Adam/v/conv2d_15/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_15/kernel*&
_output_shapes
: *
dtype0
Т
Adam/m/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/m/conv2d_15/kernel
Л
+Adam/m/conv2d_15/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_15/kernel*&
_output_shapes
: *
dtype0
Ъ
!Adam/v/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/batch_normalization_5/beta
У
5Adam/v/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_5/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/batch_normalization_5/beta
У
5Adam/m/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_5/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_5/gamma
Х
6Adam/v/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_5/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_5/gamma
Х
6Adam/m/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_5/gamma*
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
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	А*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:А*
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:А*
dtype0
}
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА* 
shared_namedense_15/kernel
v
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*!
_output_shapes
:АвА*
dtype0
u
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_17/bias
n
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_17/kernel

$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_16/bias
n
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes	
:А*
dtype0
Е
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*!
shared_nameconv2d_16/kernel
~
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*'
_output_shapes
: А*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
: *
dtype0
Д
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
: *
dtype0
в
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
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
%__inference_signature_wrapper_1382476

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
VARIABLE_VALUEbatch_normalization_5/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_5/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!batch_normalization_5/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%batch_normalization_5/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_15/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_15/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_16/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_16/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_17/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_17/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_15/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_15/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_16/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_16/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_17/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_17/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE"Adam/m/batch_normalization_5/gamma1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/batch_normalization_5/gamma1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/m/batch_normalization_5/beta1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/v/batch_normalization_5/beta1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_15/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_15/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_15/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_15/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_16/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_16/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_16/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_16/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_17/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_17/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_17/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_17/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_15/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_15/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_15/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_15/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_16/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_16/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_16/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_16/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_17/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_17/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_17/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_17/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/batch_normalization_5/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_5/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_5/beta/Read/ReadVariableOp5Adam/v/batch_normalization_5/beta/Read/ReadVariableOp+Adam/m/conv2d_15/kernel/Read/ReadVariableOp+Adam/v/conv2d_15/kernel/Read/ReadVariableOp)Adam/m/conv2d_15/bias/Read/ReadVariableOp)Adam/v/conv2d_15/bias/Read/ReadVariableOp+Adam/m/conv2d_16/kernel/Read/ReadVariableOp+Adam/v/conv2d_16/kernel/Read/ReadVariableOp)Adam/m/conv2d_16/bias/Read/ReadVariableOp)Adam/v/conv2d_16/bias/Read/ReadVariableOp+Adam/m/conv2d_17/kernel/Read/ReadVariableOp+Adam/v/conv2d_17/kernel/Read/ReadVariableOp)Adam/m/conv2d_17/bias/Read/ReadVariableOp)Adam/v/conv2d_17/bias/Read/ReadVariableOp*Adam/m/dense_15/kernel/Read/ReadVariableOp*Adam/v/dense_15/kernel/Read/ReadVariableOp(Adam/m/dense_15/bias/Read/ReadVariableOp(Adam/v/dense_15/bias/Read/ReadVariableOp*Adam/m/dense_16/kernel/Read/ReadVariableOp*Adam/v/dense_16/kernel/Read/ReadVariableOp(Adam/m/dense_16/bias/Read/ReadVariableOp(Adam/v/dense_16/bias/Read/ReadVariableOp*Adam/m/dense_17/kernel/Read/ReadVariableOp*Adam/v/dense_17/kernel/Read/ReadVariableOp(Adam/m/dense_17/bias/Read/ReadVariableOp(Adam/v/dense_17/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*?
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
 __inference__traced_save_1383546
О
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias	iterationlearning_rate"Adam/m/batch_normalization_5/gamma"Adam/v/batch_normalization_5/gamma!Adam/m/batch_normalization_5/beta!Adam/v/batch_normalization_5/betaAdam/m/conv2d_15/kernelAdam/v/conv2d_15/kernelAdam/m/conv2d_15/biasAdam/v/conv2d_15/biasAdam/m/conv2d_16/kernelAdam/v/conv2d_16/kernelAdam/m/conv2d_16/biasAdam/v/conv2d_16/biasAdam/m/conv2d_17/kernelAdam/v/conv2d_17/kernelAdam/m/conv2d_17/biasAdam/v/conv2d_17/biasAdam/m/dense_15/kernelAdam/v/dense_15/kernelAdam/m/dense_15/biasAdam/v/dense_15/biasAdam/m/dense_16/kernelAdam/v/dense_16/kernelAdam/m/dense_16/biasAdam/v/dense_16/biasAdam/m/dense_17/kernelAdam/v/dense_17/kernelAdam/m/dense_17/biasAdam/v/dense_17/biastotal_1count_1totalcount*>
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
#__inference__traced_restore_1383706щ┬
Ф

f
G__inference_dropout_17_layer_call_and_return_conditional_losses_1383346

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
■
e
G__inference_dropout_15_layer_call_and_return_conditional_losses_1381524

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
Х
i
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1383146

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
,__inference_dropout_16_layer_call_fn_1383278

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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1381687p
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
┐	
╝
__inference_loss_fn_0_1383355U
;conv2d_15_kernel_regularizer_l2loss_readvariableop_resource: 
identityИв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp╢
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp;conv2d_15_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentity$conv2d_15/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp
Х
i
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1381408

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
ъP
И	
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381871

inputs+
batch_normalization_5_1381817:+
batch_normalization_5_1381819:+
batch_normalization_5_1381821:+
batch_normalization_5_1381823:+
conv2d_15_1381826: 
conv2d_15_1381828: ,
conv2d_16_1381832: А 
conv2d_16_1381834:	А-
conv2d_17_1381838:АА 
conv2d_17_1381840:	А%
dense_15_1381846:АвА
dense_15_1381848:	А$
dense_16_1381852:
АА
dense_16_1381854:	А
identityИв-batch_normalization_5/StatefulPartitionedCallв!conv2d_15/StatefulPartitionedCallв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_16/StatefulPartitionedCallв!conv2d_17/StatefulPartitionedCallв dense_15/StatefulPartitionedCallв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв dense_16/StatefulPartitionedCallв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpв"dropout_15/StatefulPartitionedCallв"dropout_16/StatefulPartitionedCallв"dropout_17/StatefulPartitionedCall┴
lambda_5/PartitionedCallPartitionedCallinputs*
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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1381775К
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0batch_normalization_5_1381817batch_normalization_5_1381819batch_normalization_5_1381821batch_normalization_5_1381823*
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1381388п
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_15_1381826conv2d_15_1381828*
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
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1381476ї
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1381408г
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_16_1381832conv2d_16_1381834*
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
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1381494Ў
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1381420г
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_17_1381838conv2d_17_1381840*
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
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1381512Ў
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1381432∙
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
G__inference_dropout_15_layer_call_and_return_conditional_losses_1381726т
flatten_5/PartitionedCallPartitionedCall+dropout_15/StatefulPartitionedCall:output:0*
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
F__inference_flatten_5_layer_call_and_return_conditional_losses_1381532Р
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_1381846dense_15_1381848*
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
E__inference_dense_15_layer_call_and_return_conditional_losses_1381549Ц
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1381687Щ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_16_1381852dense_16_1381854*
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
E__inference_dense_16_layer_call_and_return_conditional_losses_1381577Ц
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
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
G__inference_dropout_17_layer_call_and_return_conditional_losses_1381654М
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_15_1381826*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_15_1381846*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Д
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16_1381852* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: {
IdentityIdentity+dropout_17/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А┤
NoOpNoOp.^batch_normalization_5/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
з
H
,__inference_dropout_16_layer_call_fn_1383273

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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1381560a
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
╘g
Ї
__inference_call_1319112

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: АE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	АQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ААE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	АI
4sequential_5_dense_15_matmul_readvariableop_resource:АвАD
5sequential_5_dense_15_biasadd_readvariableop_resource:	АH
4sequential_5_dense_16_matmul_readvariableop_resource:
ААD
5sequential_5_dense_16_biasadd_readvariableop_resource:	А:
'dense_17_matmul_readvariableop_resource:	А6
(dense_17_biasadd_readvariableop_resource:
identityИвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpвBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpвDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в1sequential_5/batch_normalization_5/ReadVariableOpв3sequential_5/batch_normalization_5/ReadVariableOp_1в-sequential_5/conv2d_15/BiasAdd/ReadVariableOpв,sequential_5/conv2d_15/Conv2D/ReadVariableOpв-sequential_5/conv2d_16/BiasAdd/ReadVariableOpв,sequential_5/conv2d_16/Conv2D/ReadVariableOpв-sequential_5/conv2d_17/BiasAdd/ReadVariableOpв,sequential_5/conv2d_17/Conv2D/ReadVariableOpв,sequential_5/dense_15/BiasAdd/ReadVariableOpв+sequential_5/dense_15/MatMul/ReadVariableOpв,sequential_5/dense_16/BiasAdd/ReadVariableOpв+sequential_5/dense_16/MatMul/ReadVariableOpВ
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( к
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0°
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
а
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Ж
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╚
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
л
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ё
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
б
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЗ
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╔
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
м
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
б
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЗ
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Ч
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		Аm
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  п
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         Авг
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╖
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0*(
_output_shapes
:         Ав
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╣
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0*(
_output_shapes
:         АЗ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ю
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         i
IdentityIdentitydense_17/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╓
NoOpNoOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2И
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2М
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Х
i
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1383206

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
з	
╡
__inference_loss_fn_1_1383364O
:dense_15_kernel_regularizer_l2loss_readvariableop_resource:АвА
identityИв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpп
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_15_kernel_regularizer_l2loss_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_15/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp
З
┴
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1381388

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
▐
e
G__inference_dropout_17_layer_call_and_return_conditional_losses_1383334

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
д
н
E__inference_dense_16_layer_call_and_return_conditional_losses_1381577

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpv
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
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Х
i
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1381432

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
╝
N
2__inference_max_pooling2d_15_layer_call_fn_1383141

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
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1381408Г
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
╜
о
%__inference_CNN_layer_call_fn_1382513

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
@__inference_CNN_layer_call_and_return_conditional_losses_1382117o
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
З
┴
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1383112

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
С
В
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1381512

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
ч
═
"__inference__wrapped_model_1381335
input_1
cnn_1381301:
cnn_1381303:
cnn_1381305:
cnn_1381307:%
cnn_1381309: 
cnn_1381311: &
cnn_1381313: А
cnn_1381315:	А'
cnn_1381317:АА
cnn_1381319:	А 
cnn_1381321:АвА
cnn_1381323:	А
cnn_1381325:
АА
cnn_1381327:	А
cnn_1381329:	А
cnn_1381331:
identityИвCNN/StatefulPartitionedCallК
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn_1381301cnn_1381303cnn_1381305cnn_1381307cnn_1381309cnn_1381311cnn_1381313cnn_1381315cnn_1381317cnn_1381319cnn_1381321cnn_1381323cnn_1381325cnn_1381327cnn_1381329cnn_1381331*
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
__inference_call_1319112s
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
┌
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_1383050

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
┐
F
*__inference_lambda_5_layer_call_fn_1383034

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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1381775h
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
∙
e
,__inference_dropout_17_layer_call_fn_1383329

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
G__inference_dropout_17_layer_call_and_return_conditional_losses_1381654p
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
е

ў
E__inference_dense_17_layer_call_and_return_conditional_losses_1382098

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
═
Э
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1383094

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
╝
N
2__inference_max_pooling2d_16_layer_call_fn_1383171

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
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1381420Г
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
╛
п
%__inference_CNN_layer_call_fn_1382323
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
@__inference_CNN_layer_call_and_return_conditional_losses_1382251o
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
д
н
E__inference_dense_16_layer_call_and_return_conditional_losses_1383319

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpv
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
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
в
п
%__inference_signature_wrapper_1382476
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
"__inference__wrapped_model_1381335o
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
┌
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_1381450

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
е

ў
E__inference_dense_17_layer_call_and_return_conditional_losses_1383024

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
ю
а
+__inference_conv2d_15_layer_call_fn_1383121

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
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1381476w
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
й
о
E__inference_dense_15_layer_call_and_return_conditional_losses_1381549

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpw
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
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
°
■
.__inference_sequential_5_layer_call_fn_1382796

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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381603p
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
ЮL
Щ
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381603

inputs+
batch_normalization_5_1381452:+
batch_normalization_5_1381454:+
batch_normalization_5_1381456:+
batch_normalization_5_1381458:+
conv2d_15_1381477: 
conv2d_15_1381479: ,
conv2d_16_1381495: А 
conv2d_16_1381497:	А-
conv2d_17_1381513:АА 
conv2d_17_1381515:	А%
dense_15_1381550:АвА
dense_15_1381552:	А$
dense_16_1381578:
АА
dense_16_1381580:	А
identityИв-batch_normalization_5/StatefulPartitionedCallв!conv2d_15/StatefulPartitionedCallв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_16/StatefulPartitionedCallв!conv2d_17/StatefulPartitionedCallв dense_15/StatefulPartitionedCallв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв dense_16/StatefulPartitionedCallв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp┴
lambda_5/PartitionedCallPartitionedCallinputs*
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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1381450М
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0batch_normalization_5_1381452batch_normalization_5_1381454batch_normalization_5_1381456batch_normalization_5_1381458*
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1381357п
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_15_1381477conv2d_15_1381479*
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
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1381476ї
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1381408г
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_16_1381495conv2d_16_1381497*
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
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1381494Ў
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1381420г
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_17_1381513conv2d_17_1381515*
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
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1381512Ў
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1381432щ
dropout_15/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
G__inference_dropout_15_layer_call_and_return_conditional_losses_1381524┌
flatten_5/PartitionedCallPartitionedCall#dropout_15/PartitionedCall:output:0*
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
F__inference_flatten_5_layer_call_and_return_conditional_losses_1381532Р
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_1381550dense_15_1381552*
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
E__inference_dense_15_layer_call_and_return_conditional_losses_1381549с
dropout_16/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1381560С
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_16_1381578dense_16_1381580*
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
E__inference_dense_16_layer_call_and_return_conditional_losses_1381577с
dropout_17/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
G__inference_dropout_17_layer_call_and_return_conditional_losses_1381588М
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_15_1381477*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_15_1381550*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Д
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16_1381578* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity#dropout_17/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А┼
NoOpNoOp.^batch_normalization_5/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
└
п
%__inference_CNN_layer_call_fn_1382152
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
@__inference_CNN_layer_call_and_return_conditional_losses_1382117o
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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1381687

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
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1381420

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
7__inference_batch_normalization_5_layer_call_fn_1383063

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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1381357Й
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
┌
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_1383042

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
╟
Ш
*__inference_dense_17_layer_call_fn_1383013

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
E__inference_dense_17_layer_call_and_return_conditional_losses_1382098o
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
Х
i
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1383176

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
╟
H
,__inference_dropout_15_layer_call_fn_1383211

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
G__inference_dropout_15_layer_call_and_return_conditional_losses_1381524i
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
ц
я
I__inference_sequential_5_layer_call_and_return_conditional_losses_1383004

inputs;
-batch_normalization_5_readvariableop_resource:=
/batch_normalization_5_readvariableop_1_resource:L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: C
(conv2d_16_conv2d_readvariableop_resource: А8
)conv2d_16_biasadd_readvariableop_resource:	АD
(conv2d_17_conv2d_readvariableop_resource:АА8
)conv2d_17_biasadd_readvariableop_resource:	А<
'dense_15_matmul_readvariableop_resource:АвА7
(dense_15_biasadd_readvariableop_resource:	А;
'dense_16_matmul_readvariableop_resource:
АА7
(dense_16_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_5/AssignNewValueв&batch_normalization_5/AssignNewValue_1в5batch_normalization_5/FusedBatchNormV3/ReadVariableOpв7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_5/ReadVariableOpв&batch_normalization_5/ReadVariableOp_1в conv2d_15/BiasAdd/ReadVariableOpвconv2d_15/Conv2D/ReadVariableOpв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв conv2d_16/BiasAdd/ReadVariableOpвconv2d_16/Conv2D/ReadVariableOpв conv2d_17/BiasAdd/ReadVariableOpвconv2d_17/Conv2D/ReadVariableOpвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpвdense_16/BiasAdd/ReadVariableOpвdense_16/MatMul/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpu
lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Р
lambda_5/strided_sliceStridedSliceinputs%lambda_5/strided_slice/stack:output:0'lambda_5/strided_slice/stack_1:output:0'lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskО
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╦
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3lambda_5/strided_slice:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Р
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╤
conv2d_15/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Ж
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK о
max_pooling2d_15/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
С
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0╔
conv2d_16/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
З
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аm
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ап
max_pooling2d_16/MaxPoolMaxPoolconv2d_16/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
Т
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╔
conv2d_17/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
З
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аm
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         Ап
max_pooling2d_17/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?Ю
dropout_15/dropout/MulMul!max_pooling2d_17/MaxPool:output:0!dropout_15/dropout/Const:output:0*
T0*0
_output_shapes
:         		Аi
dropout_15/dropout/ShapeShape!max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:л
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype0f
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=╨
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А_
dropout_15/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╚
dropout_15/dropout/SelectV2SelectV2#dropout_15/dropout/GreaterEqual:z:0dropout_15/dropout/Mul:z:0#dropout_15/dropout/Const_1:output:0*
T0*0
_output_shapes
:         		А`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  Р
flatten_5/ReshapeReshape$dropout_15/dropout/SelectV2:output:0flatten_5/Const:output:0*
T0*)
_output_shapes
:         АвЙ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Р
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         А]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Р
dropout_16/dropout/MulMuldense_15/Relu:activations:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:         Аc
dropout_16/dropout/ShapeShapedense_15/Relu:activations:0*
T0*
_output_shapes
:г
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╚
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А_
dropout_16/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    └
dropout_16/dropout/SelectV2SelectV2#dropout_16/dropout/GreaterEqual:z:0dropout_16/dropout/Mul:z:0#dropout_16/dropout/Const_1:output:0*
T0*(
_output_shapes
:         АИ
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ъ
dense_16/MatMulMatMul$dropout_16/dropout/SelectV2:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         А]
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Р
dropout_17/dropout/MulMuldense_16/Relu:activations:0!dropout_17/dropout/Const:output:0*
T0*(
_output_shapes
:         Аc
dropout_17/dropout/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:г
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0f
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?╚
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А_
dropout_17/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    └
dropout_17/dropout/SelectV2SelectV2#dropout_17/dropout/GreaterEqual:z:0dropout_17/dropout/Mul:z:0#dropout_17/dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аг
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ь
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: t
IdentityIdentity$dropout_17/dropout/SelectV2:output:0^NoOp*
T0*(
_output_shapes
:         А╩
NoOpNoOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
═
Э
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1381357

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
╗
о
%__inference_CNN_layer_call_fn_1382550

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
@__inference_CNN_layer_call_and_return_conditional_losses_1382251o
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
▌%
╗
@__inference_CNN_layer_call_and_return_conditional_losses_1382117

inputs"
sequential_5_1382058:"
sequential_5_1382060:"
sequential_5_1382062:"
sequential_5_1382064:.
sequential_5_1382066: "
sequential_5_1382068: /
sequential_5_1382070: А#
sequential_5_1382072:	А0
sequential_5_1382074:АА#
sequential_5_1382076:	А)
sequential_5_1382078:АвА#
sequential_5_1382080:	А(
sequential_5_1382082:
АА#
sequential_5_1382084:	А#
dense_17_1382099:	А
dense_17_1382101:
identityИв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpв dense_17/StatefulPartitionedCallв$sequential_5/StatefulPartitionedCallд
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputssequential_5_1382058sequential_5_1382060sequential_5_1382062sequential_5_1382064sequential_5_1382066sequential_5_1382068sequential_5_1382070sequential_5_1382072sequential_5_1382074sequential_5_1382076sequential_5_1382078sequential_5_1382080sequential_5_1382082sequential_5_1382084*
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381603Ъ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0dense_17_1382099dense_17_1382101*
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
E__inference_dense_17_layer_call_and_return_conditional_losses_1382098П
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382066*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382078*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382082* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_17/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
О
Ж
.__inference_sequential_5_layer_call_fn_1381935
lambda_5_input
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
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381871p
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
_user_specified_namelambda_5_input
з
H
,__inference_dropout_17_layer_call_fn_1383324

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
G__inference_dropout_17_layer_call_and_return_conditional_losses_1381588a
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
╢L
б
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381993
lambda_5_input+
batch_normalization_5_1381939:+
batch_normalization_5_1381941:+
batch_normalization_5_1381943:+
batch_normalization_5_1381945:+
conv2d_15_1381948: 
conv2d_15_1381950: ,
conv2d_16_1381954: А 
conv2d_16_1381956:	А-
conv2d_17_1381960:АА 
conv2d_17_1381962:	А%
dense_15_1381968:АвА
dense_15_1381970:	А$
dense_16_1381974:
АА
dense_16_1381976:	А
identityИв-batch_normalization_5/StatefulPartitionedCallв!conv2d_15/StatefulPartitionedCallв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_16/StatefulPartitionedCallв!conv2d_17/StatefulPartitionedCallв dense_15/StatefulPartitionedCallв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв dense_16/StatefulPartitionedCallв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp╔
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1381450М
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0batch_normalization_5_1381939batch_normalization_5_1381941batch_normalization_5_1381943batch_normalization_5_1381945*
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1381357п
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_15_1381948conv2d_15_1381950*
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
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1381476ї
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1381408г
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_16_1381954conv2d_16_1381956*
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
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1381494Ў
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1381420г
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_17_1381960conv2d_17_1381962*
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
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1381512Ў
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1381432щ
dropout_15/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
G__inference_dropout_15_layer_call_and_return_conditional_losses_1381524┌
flatten_5/PartitionedCallPartitionedCall#dropout_15/PartitionedCall:output:0*
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
F__inference_flatten_5_layer_call_and_return_conditional_losses_1381532Р
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_1381968dense_15_1381970*
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
E__inference_dense_15_layer_call_and_return_conditional_losses_1381549с
dropout_16/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1381560С
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_16_1381974dense_16_1381976*
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
E__inference_dense_16_layer_call_and_return_conditional_losses_1381577с
dropout_17/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
G__inference_dropout_17_layer_call_and_return_conditional_losses_1381588М
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_15_1381948*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_15_1381968*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Д
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16_1381974* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity#dropout_17/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А┼
NoOpNoOp.^batch_normalization_5/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_5_input
■
e
G__inference_dropout_15_layer_call_and_return_conditional_losses_1383221

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
╦
Ъ
*__inference_dense_16_layer_call_fn_1383304

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
E__inference_dense_16_layer_call_and_return_conditional_losses_1381577p
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
▐
e
G__inference_dropout_16_layer_call_and_return_conditional_losses_1383283

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
сe
Ї
__inference_call_1320393

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: АE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	АQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ААE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	АI
4sequential_5_dense_15_matmul_readvariableop_resource:АвАD
5sequential_5_dense_15_biasadd_readvariableop_resource:	АH
4sequential_5_dense_16_matmul_readvariableop_resource:
ААD
5sequential_5_dense_16_biasadd_readvariableop_resource:	А:
'dense_17_matmul_readvariableop_resource:	А6
(dense_17_biasadd_readvariableop_resource:
identityИвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpвBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpвDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в1sequential_5/batch_normalization_5/ReadVariableOpв3sequential_5/batch_normalization_5/ReadVariableOp_1в-sequential_5/conv2d_15/BiasAdd/ReadVariableOpв,sequential_5/conv2d_15/Conv2D/ReadVariableOpв-sequential_5/conv2d_16/BiasAdd/ReadVariableOpв,sequential_5/conv2d_16/Conv2D/ReadVariableOpв-sequential_5/conv2d_17/BiasAdd/ReadVariableOpв,sequential_5/conv2d_17/Conv2D/ReadVariableOpв,sequential_5/dense_15/BiasAdd/ReadVariableOpв+sequential_5/dense_15/MatMul/ReadVariableOpв,sequential_5/dense_16/BiasAdd/ReadVariableOpв+sequential_5/dense_16/MatMul/ReadVariableOpВ
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ╝
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_maskи
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Г
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( к
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ё
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
а
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0║
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK ~
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*'
_output_shapes
:АKK └
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
л
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0ш
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
б
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А┴
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
м
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ш
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
б
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*(
_output_shapes
:АА┴
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
П
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*(
_output_shapes
:А		Аm
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  з
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*!
_output_shapes
:ААвг
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0п
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЯ
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▒
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААu
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0* 
_output_shapes
:
ААБ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0* 
_output_shapes
:
ААв
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0▒
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЯ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▒
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААu
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0* 
_output_shapes
:
ААБ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0* 
_output_shapes
:
ААЗ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	АД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Й
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А`
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*
_output_shapes
:	Аa
IdentityIdentitydense_17/Softmax:softmax:0^NoOp*
T0*
_output_shapes
:	А╓
NoOpNoOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2И
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2М
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
╠

f
G__inference_dropout_15_layer_call_and_return_conditional_losses_1381726

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
Р
Ж
.__inference_sequential_5_layer_call_fn_1381634
lambda_5_input
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
StatefulPartitionedCallStatefulPartitionedCalllambda_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381603p
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
_user_specified_namelambda_5_input
┬z
╣
@__inference_CNN_layer_call_and_return_conditional_losses_1382634

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: АE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	АQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ААE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	АI
4sequential_5_dense_15_matmul_readvariableop_resource:АвАD
5sequential_5_dense_15_biasadd_readvariableop_resource:	АH
4sequential_5_dense_16_matmul_readvariableop_resource:
ААD
5sequential_5_dense_16_biasadd_readvariableop_resource:	А:
'dense_17_matmul_readvariableop_resource:	А6
(dense_17_biasadd_readvariableop_resource:
identityИв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpвBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpвDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в1sequential_5/batch_normalization_5/ReadVariableOpв3sequential_5/batch_normalization_5/ReadVariableOp_1в-sequential_5/conv2d_15/BiasAdd/ReadVariableOpв,sequential_5/conv2d_15/Conv2D/ReadVariableOpв-sequential_5/conv2d_16/BiasAdd/ReadVariableOpв,sequential_5/conv2d_16/Conv2D/ReadVariableOpв-sequential_5/conv2d_17/BiasAdd/ReadVariableOpв,sequential_5/conv2d_17/Conv2D/ReadVariableOpв,sequential_5/dense_15/BiasAdd/ReadVariableOpв+sequential_5/dense_15/MatMul/ReadVariableOpв,sequential_5/dense_16/BiasAdd/ReadVariableOpв+sequential_5/dense_16/MatMul/ReadVariableOpВ
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( к
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0°
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
а
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Ж
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╚
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
л
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ё
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
б
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЗ
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╔
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
м
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
б
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЗ
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Ч
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		Аm
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  п
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         Авг
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╖
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0*(
_output_shapes
:         Ав
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╣
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0*(
_output_shapes
:         АЗ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ю
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         ░
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: й
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: и
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_17/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         є
NoOpNoOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2И
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2М
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▐
e
G__inference_dropout_16_layer_call_and_return_conditional_losses_1381560

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
С
В
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1383196

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
╘g
Ї
__inference_call_1320465

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: АE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	АQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ААE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	АI
4sequential_5_dense_15_matmul_readvariableop_resource:АвАD
5sequential_5_dense_15_biasadd_readvariableop_resource:	АH
4sequential_5_dense_16_matmul_readvariableop_resource:
ААD
5sequential_5_dense_16_biasadd_readvariableop_resource:	А:
'dense_17_matmul_readvariableop_resource:	А6
(dense_17_biasadd_readvariableop_resource:
identityИвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpвBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpвDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в1sequential_5/batch_normalization_5/ReadVariableOpв3sequential_5/batch_normalization_5/ReadVariableOp_1в-sequential_5/conv2d_15/BiasAdd/ReadVariableOpв,sequential_5/conv2d_15/Conv2D/ReadVariableOpв-sequential_5/conv2d_16/BiasAdd/ReadVariableOpв,sequential_5/conv2d_16/Conv2D/ReadVariableOpв-sequential_5/conv2d_17/BiasAdd/ReadVariableOpв,sequential_5/conv2d_17/Conv2D/ReadVariableOpв,sequential_5/dense_15/BiasAdd/ReadVariableOpв+sequential_5/dense_15/MatMul/ReadVariableOpв,sequential_5/dense_16/BiasAdd/ReadVariableOpв+sequential_5/dense_16/MatMul/ReadVariableOpВ
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( к
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0°
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
а
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Ж
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╚
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
л
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ё
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
б
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЗ
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╔
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
м
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
б
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЗ
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Ч
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		Аm
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  п
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         Авг
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╖
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0*(
_output_shapes
:         Ав
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╣
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         АЙ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0*(
_output_shapes
:         АЗ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ю
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         i
IdentityIdentitydense_17/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╓
NoOpNoOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2И
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2М
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╝╤
Ж 
#__inference__traced_restore_1383706
file_prefix:
,assignvariableop_batch_normalization_5_gamma:;
-assignvariableop_1_batch_normalization_5_beta:B
4assignvariableop_2_batch_normalization_5_moving_mean:F
8assignvariableop_3_batch_normalization_5_moving_variance:=
#assignvariableop_4_conv2d_15_kernel: /
!assignvariableop_5_conv2d_15_bias: >
#assignvariableop_6_conv2d_16_kernel: А0
!assignvariableop_7_conv2d_16_bias:	А?
#assignvariableop_8_conv2d_17_kernel:АА0
!assignvariableop_9_conv2d_17_bias:	А8
#assignvariableop_10_dense_15_kernel:АвА0
!assignvariableop_11_dense_15_bias:	А7
#assignvariableop_12_dense_16_kernel:
АА0
!assignvariableop_13_dense_16_bias:	А6
#assignvariableop_14_dense_17_kernel:	А/
!assignvariableop_15_dense_17_bias:'
assignvariableop_16_iteration:	 +
!assignvariableop_17_learning_rate: D
6assignvariableop_18_adam_m_batch_normalization_5_gamma:D
6assignvariableop_19_adam_v_batch_normalization_5_gamma:C
5assignvariableop_20_adam_m_batch_normalization_5_beta:C
5assignvariableop_21_adam_v_batch_normalization_5_beta:E
+assignvariableop_22_adam_m_conv2d_15_kernel: E
+assignvariableop_23_adam_v_conv2d_15_kernel: 7
)assignvariableop_24_adam_m_conv2d_15_bias: 7
)assignvariableop_25_adam_v_conv2d_15_bias: F
+assignvariableop_26_adam_m_conv2d_16_kernel: АF
+assignvariableop_27_adam_v_conv2d_16_kernel: А8
)assignvariableop_28_adam_m_conv2d_16_bias:	А8
)assignvariableop_29_adam_v_conv2d_16_bias:	АG
+assignvariableop_30_adam_m_conv2d_17_kernel:ААG
+assignvariableop_31_adam_v_conv2d_17_kernel:АА8
)assignvariableop_32_adam_m_conv2d_17_bias:	А8
)assignvariableop_33_adam_v_conv2d_17_bias:	А?
*assignvariableop_34_adam_m_dense_15_kernel:АвА?
*assignvariableop_35_adam_v_dense_15_kernel:АвА7
(assignvariableop_36_adam_m_dense_15_bias:	А7
(assignvariableop_37_adam_v_dense_15_bias:	А>
*assignvariableop_38_adam_m_dense_16_kernel:
АА>
*assignvariableop_39_adam_v_dense_16_kernel:
АА7
(assignvariableop_40_adam_m_dense_16_bias:	А7
(assignvariableop_41_adam_v_dense_16_bias:	А=
*assignvariableop_42_adam_m_dense_17_kernel:	А=
*assignvariableop_43_adam_v_dense_17_kernel:	А6
(assignvariableop_44_adam_m_dense_17_bias:6
(assignvariableop_45_adam_v_dense_17_bias:%
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
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_5_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_5_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_5_moving_meanIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_5_moving_varianceIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_15_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_15_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_16_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_16_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_17_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_17_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_15_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_15_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_16_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_16_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_17_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_17_biasIdentity_15:output:0"/device:CPU:0*&
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
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_m_batch_normalization_5_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_v_batch_normalization_5_gammaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_m_batch_normalization_5_betaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_v_batch_normalization_5_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_m_conv2d_15_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_v_conv2d_15_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_m_conv2d_15_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_v_conv2d_15_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_m_conv2d_16_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_v_conv2d_16_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_m_conv2d_16_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_v_conv2d_16_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_m_conv2d_17_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_v_conv2d_17_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_m_conv2d_17_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_v_conv2d_17_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_m_dense_15_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_v_dense_15_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_m_dense_15_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_v_dense_15_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_dense_16_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_dense_16_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_dense_16_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_dense_16_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_m_dense_17_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_v_dense_17_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_m_dense_17_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_v_dense_17_biasIdentity_45:output:0"/device:CPU:0*&
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
┌
a
E__inference_lambda_5_layer_call_and_return_conditional_losses_1381775

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
Ў
■
.__inference_sequential_5_layer_call_fn_1382829

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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381871p
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
╒_
└
 __inference__traced_save_1383546
file_prefix:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_batch_normalization_5_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_5_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_5_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_5_beta_read_readvariableop6
2savev2_adam_m_conv2d_15_kernel_read_readvariableop6
2savev2_adam_v_conv2d_15_kernel_read_readvariableop4
0savev2_adam_m_conv2d_15_bias_read_readvariableop4
0savev2_adam_v_conv2d_15_bias_read_readvariableop6
2savev2_adam_m_conv2d_16_kernel_read_readvariableop6
2savev2_adam_v_conv2d_16_kernel_read_readvariableop4
0savev2_adam_m_conv2d_16_bias_read_readvariableop4
0savev2_adam_v_conv2d_16_bias_read_readvariableop6
2savev2_adam_m_conv2d_17_kernel_read_readvariableop6
2savev2_adam_v_conv2d_17_kernel_read_readvariableop4
0savev2_adam_m_conv2d_17_bias_read_readvariableop4
0savev2_adam_v_conv2d_17_bias_read_readvariableop5
1savev2_adam_m_dense_15_kernel_read_readvariableop5
1savev2_adam_v_dense_15_kernel_read_readvariableop3
/savev2_adam_m_dense_15_bias_read_readvariableop3
/savev2_adam_v_dense_15_bias_read_readvariableop5
1savev2_adam_m_dense_16_kernel_read_readvariableop5
1savev2_adam_v_dense_16_kernel_read_readvariableop3
/savev2_adam_m_dense_16_bias_read_readvariableop3
/savev2_adam_v_dense_16_bias_read_readvariableop5
1savev2_adam_m_dense_17_kernel_read_readvariableop5
1savev2_adam_v_dense_17_kernel_read_readvariableop3
/savev2_adam_m_dense_17_bias_read_readvariableop3
/savev2_adam_v_dense_17_bias_read_readvariableop&
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_batch_normalization_5_gamma_read_readvariableop=savev2_adam_v_batch_normalization_5_gamma_read_readvariableop<savev2_adam_m_batch_normalization_5_beta_read_readvariableop<savev2_adam_v_batch_normalization_5_beta_read_readvariableop2savev2_adam_m_conv2d_15_kernel_read_readvariableop2savev2_adam_v_conv2d_15_kernel_read_readvariableop0savev2_adam_m_conv2d_15_bias_read_readvariableop0savev2_adam_v_conv2d_15_bias_read_readvariableop2savev2_adam_m_conv2d_16_kernel_read_readvariableop2savev2_adam_v_conv2d_16_kernel_read_readvariableop0savev2_adam_m_conv2d_16_bias_read_readvariableop0savev2_adam_v_conv2d_16_bias_read_readvariableop2savev2_adam_m_conv2d_17_kernel_read_readvariableop2savev2_adam_v_conv2d_17_kernel_read_readvariableop0savev2_adam_m_conv2d_17_bias_read_readvariableop0savev2_adam_v_conv2d_17_bias_read_readvariableop1savev2_adam_m_dense_15_kernel_read_readvariableop1savev2_adam_v_dense_15_kernel_read_readvariableop/savev2_adam_m_dense_15_bias_read_readvariableop/savev2_adam_v_dense_15_bias_read_readvariableop1savev2_adam_m_dense_16_kernel_read_readvariableop1savev2_adam_v_dense_16_kernel_read_readvariableop/savev2_adam_m_dense_16_bias_read_readvariableop/savev2_adam_v_dense_16_bias_read_readvariableop1savev2_adam_m_dense_17_kernel_read_readvariableop1savev2_adam_v_dense_17_kernel_read_readvariableop/savev2_adam_m_dense_17_bias_read_readvariableop/savev2_adam_v_dense_17_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
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
ї
г
+__inference_conv2d_17_layer_call_fn_1383185

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
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1381512x
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
▐
e
G__inference_dropout_17_layer_call_and_return_conditional_losses_1381588

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
й
о
E__inference_dense_15_layer_call_and_return_conditional_losses_1383268

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpw
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
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ал
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
╠
b
F__inference_flatten_5_layer_call_and_return_conditional_losses_1383244

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
ВQ
Р	
I__inference_sequential_5_layer_call_and_return_conditional_losses_1382051
lambda_5_input+
batch_normalization_5_1381997:+
batch_normalization_5_1381999:+
batch_normalization_5_1382001:+
batch_normalization_5_1382003:+
conv2d_15_1382006: 
conv2d_15_1382008: ,
conv2d_16_1382012: А 
conv2d_16_1382014:	А-
conv2d_17_1382018:АА 
conv2d_17_1382020:	А%
dense_15_1382026:АвА
dense_15_1382028:	А$
dense_16_1382032:
АА
dense_16_1382034:	А
identityИв-batch_normalization_5/StatefulPartitionedCallв!conv2d_15/StatefulPartitionedCallв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв!conv2d_16/StatefulPartitionedCallв!conv2d_17/StatefulPartitionedCallв dense_15/StatefulPartitionedCallв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв dense_16/StatefulPartitionedCallв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpв"dropout_15/StatefulPartitionedCallв"dropout_16/StatefulPartitionedCallв"dropout_17/StatefulPartitionedCall╔
lambda_5/PartitionedCallPartitionedCalllambda_5_input*
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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1381775К
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall!lambda_5/PartitionedCall:output:0batch_normalization_5_1381997batch_normalization_5_1381999batch_normalization_5_1382001batch_normalization_5_1382003*
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1381388п
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv2d_15_1382006conv2d_15_1382008*
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
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1381476ї
 max_pooling2d_15/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1381408г
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_15/PartitionedCall:output:0conv2d_16_1382012conv2d_16_1382014*
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
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1381494Ў
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1381420г
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_17_1382018conv2d_17_1382020*
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
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1381512Ў
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1381432∙
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
G__inference_dropout_15_layer_call_and_return_conditional_losses_1381726т
flatten_5/PartitionedCallPartitionedCall+dropout_15/StatefulPartitionedCall:output:0*
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
F__inference_flatten_5_layer_call_and_return_conditional_losses_1381532Р
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_1382026dense_15_1382028*
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
E__inference_dense_15_layer_call_and_return_conditional_losses_1381549Ц
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0#^dropout_15/StatefulPartitionedCall*
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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1381687Щ
 dense_16/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_16_1382032dense_16_1382034*
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
E__inference_dense_16_layer_call_and_return_conditional_losses_1381577Ц
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
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
G__inference_dropout_17_layer_call_and_return_conditional_losses_1381654М
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_15_1382006*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Е
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_15_1382026*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Д
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_16_1382032* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: {
IdentityIdentity+dropout_17/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А┤
NoOpNoOp.^batch_normalization_5/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp#^dropout_15/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_5_input
Н
Б
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1381494

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
е	
┤
__inference_loss_fn_2_1383373N
:dense_16_kernel_regularizer_l2loss_readvariableop_resource:
АА
identityИв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpо
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_16_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_16/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp
╝
N
2__inference_max_pooling2d_17_layer_call_fn_1383201

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
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1381432Г
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
╥Я
г
@__inference_CNN_layer_call_and_return_conditional_losses_1382739

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: АE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	АQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ААE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	АI
4sequential_5_dense_15_matmul_readvariableop_resource:АвАD
5sequential_5_dense_15_biasadd_readvariableop_resource:	АH
4sequential_5_dense_16_matmul_readvariableop_resource:
ААD
5sequential_5_dense_16_biasadd_readvariableop_resource:	А:
'dense_17_matmul_readvariableop_resource:	А6
(dense_17_biasadd_readvariableop_resource:
identityИв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpв1sequential_5/batch_normalization_5/AssignNewValueв3sequential_5/batch_normalization_5/AssignNewValue_1вBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpвDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в1sequential_5/batch_normalization_5/ReadVariableOpв3sequential_5/batch_normalization_5/ReadVariableOp_1в-sequential_5/conv2d_15/BiasAdd/ReadVariableOpв,sequential_5/conv2d_15/Conv2D/ReadVariableOpв-sequential_5/conv2d_16/BiasAdd/ReadVariableOpв,sequential_5/conv2d_16/Conv2D/ReadVariableOpв-sequential_5/conv2d_17/BiasAdd/ReadVariableOpв,sequential_5/conv2d_17/Conv2D/ReadVariableOpв,sequential_5/dense_15/BiasAdd/ReadVariableOpв+sequential_5/dense_15/MatMul/ReadVariableOpв,sequential_5/dense_16/BiasAdd/ReadVariableOpв+sequential_5/dense_16/MatMul/ReadVariableOpВ
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Щ
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╥
1sequential_5/batch_normalization_5/AssignNewValueAssignVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource@sequential_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0C^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(▄
3sequential_5/batch_normalization_5/AssignNewValue_1AssignVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceDsequential_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0E^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(к
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0°
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
а
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┬
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Ж
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╚
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
л
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0Ё
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
б
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЗ
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╔
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
м
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ё
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
б
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0├
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЗ
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         А╔
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
j
%sequential_5/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?┼
#sequential_5/dropout_15/dropout/MulMul.sequential_5/max_pooling2d_17/MaxPool:output:0.sequential_5/dropout_15/dropout/Const:output:0*
T0*0
_output_shapes
:         		АГ
%sequential_5/dropout_15/dropout/ShapeShape.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:┼
<sequential_5/dropout_15/dropout/random_uniform/RandomUniformRandomUniform.sequential_5/dropout_15/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype0s
.sequential_5/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=ў
,sequential_5/dropout_15/dropout/GreaterEqualGreaterEqualEsequential_5/dropout_15/dropout/random_uniform/RandomUniform:output:07sequential_5/dropout_15/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		Аl
'sequential_5/dropout_15/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    №
(sequential_5/dropout_15/dropout/SelectV2SelectV20sequential_5/dropout_15/dropout/GreaterEqual:z:0'sequential_5/dropout_15/dropout/Mul:z:00sequential_5/dropout_15/dropout/Const_1:output:0*
T0*0
_output_shapes
:         		Аm
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  ╖
sequential_5/flatten_5/ReshapeReshape1sequential_5/dropout_15/dropout/SelectV2:output:0%sequential_5/flatten_5/Const:output:0*
T0*)
_output_shapes
:         Авг
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╖
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         Аj
%sequential_5/dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @╖
#sequential_5/dropout_16/dropout/MulMul(sequential_5/dense_15/Relu:activations:0.sequential_5/dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:         А}
%sequential_5/dropout_16/dropout/ShapeShape(sequential_5/dense_15/Relu:activations:0*
T0*
_output_shapes
:╜
<sequential_5/dropout_16/dropout/random_uniform/RandomUniformRandomUniform.sequential_5/dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0s
.sequential_5/dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?я
,sequential_5/dropout_16/dropout/GreaterEqualGreaterEqualEsequential_5/dropout_16/dropout/random_uniform/RandomUniform:output:07sequential_5/dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аl
'sequential_5/dropout_16/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ї
(sequential_5/dropout_16/dropout/SelectV2SelectV20sequential_5/dropout_16/dropout/GreaterEqual:z:0'sequential_5/dropout_16/dropout/Mul:z:00sequential_5/dropout_16/dropout/Const_1:output:0*
T0*(
_output_shapes
:         Ав
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0┴
sequential_5/dense_16/MatMulMatMul1sequential_5/dropout_16/dropout/SelectV2:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╣
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А}
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         Аj
%sequential_5/dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @╖
#sequential_5/dropout_17/dropout/MulMul(sequential_5/dense_16/Relu:activations:0.sequential_5/dropout_17/dropout/Const:output:0*
T0*(
_output_shapes
:         А}
%sequential_5/dropout_17/dropout/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:╜
<sequential_5/dropout_17/dropout/random_uniform/RandomUniformRandomUniform.sequential_5/dropout_17/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0s
.sequential_5/dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?я
,sequential_5/dropout_17/dropout/GreaterEqualGreaterEqualEsequential_5/dropout_17/dropout/random_uniform/RandomUniform:output:07sequential_5/dropout_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аl
'sequential_5/dropout_17/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ї
(sequential_5/dropout_17/dropout/SelectV2SelectV20sequential_5/dropout_17/dropout/GreaterEqual:z:0'sequential_5/dropout_17/dropout/Mul:z:00sequential_5/dropout_17/dropout/Const_1:output:0*
T0*(
_output_shapes
:         АЗ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0ж
dense_17/MatMulMatMul1sequential_5/dropout_17/dropout/SelectV2:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:         ░
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: й
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: и
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_17/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ▌
NoOpNoOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp2^sequential_5/batch_normalization_5/AssignNewValue4^sequential_5/batch_normalization_5/AssignNewValue_1C^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2f
1sequential_5/batch_normalization_5/AssignNewValue1sequential_5/batch_normalization_5/AssignNewValue2j
3sequential_5/batch_normalization_5/AssignNewValue_13sequential_5/batch_normalization_5/AssignNewValue_12И
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2М
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▐%
╝
@__inference_CNN_layer_call_and_return_conditional_losses_1382423
input_1"
sequential_5_1382376:"
sequential_5_1382378:"
sequential_5_1382380:"
sequential_5_1382382:.
sequential_5_1382384: "
sequential_5_1382386: /
sequential_5_1382388: А#
sequential_5_1382390:	А0
sequential_5_1382392:АА#
sequential_5_1382394:	А)
sequential_5_1382396:АвА#
sequential_5_1382398:	А(
sequential_5_1382400:
АА#
sequential_5_1382402:	А#
dense_17_1382405:	А
dense_17_1382407:
identityИв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpв dense_17/StatefulPartitionedCallв$sequential_5/StatefulPartitionedCallг
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_5_1382376sequential_5_1382378sequential_5_1382380sequential_5_1382382sequential_5_1382384sequential_5_1382386sequential_5_1382388sequential_5_1382390sequential_5_1382392sequential_5_1382394sequential_5_1382396sequential_5_1382398sequential_5_1382400sequential_5_1382402*
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381871Ъ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0dense_17_1382405dense_17_1382407*
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
E__inference_dense_17_layer_call_and_return_conditional_losses_1382098П
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382384*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382396*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382400* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_17/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
█%
╗
@__inference_CNN_layer_call_and_return_conditional_losses_1382251

inputs"
sequential_5_1382204:"
sequential_5_1382206:"
sequential_5_1382208:"
sequential_5_1382210:.
sequential_5_1382212: "
sequential_5_1382214: /
sequential_5_1382216: А#
sequential_5_1382218:	А0
sequential_5_1382220:АА#
sequential_5_1382222:	А)
sequential_5_1382224:АвА#
sequential_5_1382226:	А(
sequential_5_1382228:
АА#
sequential_5_1382230:	А#
dense_17_1382233:	А
dense_17_1382235:
identityИв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpв dense_17/StatefulPartitionedCallв$sequential_5/StatefulPartitionedCallв
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinputssequential_5_1382204sequential_5_1382206sequential_5_1382208sequential_5_1382210sequential_5_1382212sequential_5_1382214sequential_5_1382216sequential_5_1382218sequential_5_1382220sequential_5_1382222sequential_5_1382224sequential_5_1382226sequential_5_1382228sequential_5_1382230*
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381871Ъ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0dense_17_1382233dense_17_1382235*
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
E__inference_dense_17_layer_call_and_return_conditional_losses_1382098П
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382212*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382224*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382228* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_17/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ф

f
G__inference_dropout_17_layer_call_and_return_conditional_losses_1381654

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
┐
F
*__inference_lambda_5_layer_call_fn_1383029

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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1381450h
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
С	
╥
7__inference_batch_normalization_5_layer_call_fn_1383076

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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1381388Й
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
Т
┤
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1383136

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp|
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
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         KK м
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
сe
Ї
__inference_call_1320321

inputsH
:sequential_5_batch_normalization_5_readvariableop_resource:J
<sequential_5_batch_normalization_5_readvariableop_1_resource:Y
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:[
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_5_conv2d_15_conv2d_readvariableop_resource: D
6sequential_5_conv2d_15_biasadd_readvariableop_resource: P
5sequential_5_conv2d_16_conv2d_readvariableop_resource: АE
6sequential_5_conv2d_16_biasadd_readvariableop_resource:	АQ
5sequential_5_conv2d_17_conv2d_readvariableop_resource:ААE
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	АI
4sequential_5_dense_15_matmul_readvariableop_resource:АвАD
5sequential_5_dense_15_biasadd_readvariableop_resource:	АH
4sequential_5_dense_16_matmul_readvariableop_resource:
ААD
5sequential_5_dense_16_biasadd_readvariableop_resource:	А:
'dense_17_matmul_readvariableop_resource:	А6
(dense_17_biasadd_readvariableop_resource:
identityИвdense_17/BiasAdd/ReadVariableOpвdense_17/MatMul/ReadVariableOpвBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpвDsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в1sequential_5/batch_normalization_5/ReadVariableOpв3sequential_5/batch_normalization_5/ReadVariableOp_1в-sequential_5/conv2d_15/BiasAdd/ReadVariableOpв,sequential_5/conv2d_15/Conv2D/ReadVariableOpв-sequential_5/conv2d_16/BiasAdd/ReadVariableOpв,sequential_5/conv2d_16/Conv2D/ReadVariableOpв-sequential_5/conv2d_17/BiasAdd/ReadVariableOpв,sequential_5/conv2d_17/Conv2D/ReadVariableOpв,sequential_5/dense_15/BiasAdd/ReadVariableOpв+sequential_5/dense_15/MatMul/ReadVariableOpв,sequential_5/dense_16/BiasAdd/ReadVariableOpв+sequential_5/dense_16/MatMul/ReadVariableOpВ
)sequential_5/lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_5/lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ╝
#sequential_5/lambda_5/strided_sliceStridedSliceinputs2sequential_5/lambda_5/strided_slice/stack:output:04sequential_5/lambda_5/strided_slice/stack_1:output:04sequential_5/lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_maskи
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Г
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3,sequential_5/lambda_5/strided_slice:output:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( к
,sequential_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ё
sequential_5/conv2d_15/Conv2DConv2D7sequential_5/batch_normalization_5/FusedBatchNormV3:y:04sequential_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
а
-sequential_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0║
sequential_5/conv2d_15/BiasAddBiasAdd&sequential_5/conv2d_15/Conv2D:output:05sequential_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK ~
sequential_5/conv2d_15/ReluRelu'sequential_5/conv2d_15/BiasAdd:output:0*
T0*'
_output_shapes
:АKK └
%sequential_5/max_pooling2d_15/MaxPoolMaxPool)sequential_5/conv2d_15/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
л
,sequential_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0ш
sequential_5/conv2d_16/Conv2DConv2D.sequential_5/max_pooling2d_15/MaxPool:output:04sequential_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
б
-sequential_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
sequential_5/conv2d_16/BiasAddBiasAdd&sequential_5/conv2d_16/Conv2D:output:05sequential_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А
sequential_5/conv2d_16/ReluRelu'sequential_5/conv2d_16/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А┴
%sequential_5/max_pooling2d_16/MaxPoolMaxPool)sequential_5/conv2d_16/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
м
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0ш
sequential_5/conv2d_17/Conv2DConv2D.sequential_5/max_pooling2d_16/MaxPool:output:04sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
б
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╗
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА
sequential_5/conv2d_17/ReluRelu'sequential_5/conv2d_17/BiasAdd:output:0*
T0*(
_output_shapes
:АА┴
%sequential_5/max_pooling2d_17/MaxPoolMaxPool)sequential_5/conv2d_17/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
П
 sequential_5/dropout_15/IdentityIdentity.sequential_5/max_pooling2d_17/MaxPool:output:0*
T0*(
_output_shapes
:А		Аm
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  з
sequential_5/flatten_5/ReshapeReshape)sequential_5/dropout_15/Identity:output:0%sequential_5/flatten_5/Const:output:0*
T0*!
_output_shapes
:ААвг
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0п
sequential_5/dense_15/MatMulMatMul'sequential_5/flatten_5/Reshape:output:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЯ
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▒
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААu
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0* 
_output_shapes
:
ААБ
 sequential_5/dropout_16/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0* 
_output_shapes
:
ААв
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0▒
sequential_5/dense_16/MatMulMatMul)sequential_5/dropout_16/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЯ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▒
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААu
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0* 
_output_shapes
:
ААБ
 sequential_5/dropout_17/IdentityIdentity(sequential_5/dense_16/Relu:activations:0*
T0* 
_output_shapes
:
ААЗ
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
dense_17/MatMulMatMul)sequential_5/dropout_17/Identity:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	АД
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Й
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А`
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*
_output_shapes
:	Аa
IdentityIdentitydense_17/Softmax:softmax:0^NoOp*
T0*
_output_shapes
:	А╓
NoOpNoOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOpC^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batch_normalization_5/ReadVariableOp4^sequential_5/batch_normalization_5/ReadVariableOp_1.^sequential_5/conv2d_15/BiasAdd/ReadVariableOp-^sequential_5/conv2d_15/Conv2D/ReadVariableOp.^sequential_5/conv2d_16/BiasAdd/ReadVariableOp-^sequential_5/conv2d_16/Conv2D/ReadVariableOp.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2И
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2М
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batch_normalization_5/ReadVariableOp1sequential_5/batch_normalization_5/ReadVariableOp2j
3sequential_5/batch_normalization_5/ReadVariableOp_13sequential_5/batch_normalization_5/ReadVariableOp_12^
-sequential_5/conv2d_15/BiasAdd/ReadVariableOp-sequential_5/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_15/Conv2D/ReadVariableOp,sequential_5/conv2d_15/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_16/BiasAdd/ReadVariableOp-sequential_5/conv2d_16/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_16/Conv2D/ReadVariableOp,sequential_5/conv2d_16/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
Н
Б
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1383166

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
р%
╝
@__inference_CNN_layer_call_and_return_conditional_losses_1382373
input_1"
sequential_5_1382326:"
sequential_5_1382328:"
sequential_5_1382330:"
sequential_5_1382332:.
sequential_5_1382334: "
sequential_5_1382336: /
sequential_5_1382338: А#
sequential_5_1382340:	А0
sequential_5_1382342:АА#
sequential_5_1382344:	А)
sequential_5_1382346:АвА#
sequential_5_1382348:	А(
sequential_5_1382350:
АА#
sequential_5_1382352:	А#
dense_17_1382355:	А
dense_17_1382357:
identityИв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpв dense_17/StatefulPartitionedCallв$sequential_5/StatefulPartitionedCallе
$sequential_5/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_5_1382326sequential_5_1382328sequential_5_1382330sequential_5_1382332sequential_5_1382334sequential_5_1382336sequential_5_1382338sequential_5_1382340sequential_5_1382342sequential_5_1382344sequential_5_1382346sequential_5_1382348sequential_5_1382350sequential_5_1382352*
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381603Ъ
 dense_17/StatefulPartitionedCallStatefulPartitionedCall-sequential_5/StatefulPartitionedCall:output:0dense_17_1382355dense_17_1382357*
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
E__inference_dense_17_layer_call_and_return_conditional_losses_1382098П
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382334*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Й
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382346*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_5_1382350* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_17/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
╖
G
+__inference_flatten_5_layer_call_fn_1383238

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
F__inference_flatten_5_layer_call_and_return_conditional_losses_1381532b
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
╬
Ы
*__inference_dense_15_layer_call_fn_1383253

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
E__inference_dense_15_layer_call_and_return_conditional_losses_1381549p
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
Ф

f
G__inference_dropout_16_layer_call_and_return_conditional_losses_1383295

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
Є
в
+__inference_conv2d_16_layer_call_fn_1383155

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
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1381494x
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
э`
Я
I__inference_sequential_5_layer_call_and_return_conditional_losses_1382906

inputs;
-batch_normalization_5_readvariableop_resource:=
/batch_normalization_5_readvariableop_1_resource:L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_15_conv2d_readvariableop_resource: 7
)conv2d_15_biasadd_readvariableop_resource: C
(conv2d_16_conv2d_readvariableop_resource: А8
)conv2d_16_biasadd_readvariableop_resource:	АD
(conv2d_17_conv2d_readvariableop_resource:АА8
)conv2d_17_biasadd_readvariableop_resource:	А<
'dense_15_matmul_readvariableop_resource:АвА7
(dense_15_biasadd_readvariableop_resource:	А;
'dense_16_matmul_readvariableop_resource:
АА7
(dense_16_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_5/FusedBatchNormV3/ReadVariableOpв7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_5/ReadVariableOpв&batch_normalization_5/ReadVariableOp_1в conv2d_15/BiasAdd/ReadVariableOpвconv2d_15/Conv2D/ReadVariableOpв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpв conv2d_16/BiasAdd/ReadVariableOpвconv2d_16/Conv2D/ReadVariableOpв conv2d_17/BiasAdd/ReadVariableOpвconv2d_17/Conv2D/ReadVariableOpвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpв1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpвdense_16/BiasAdd/ReadVariableOpвdense_16/MatMul/ReadVariableOpв1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpu
lambda_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Р
lambda_5/strided_sliceStridedSliceinputs%lambda_5/strided_slice/stack:output:0'lambda_5/strided_slice/stack_1:output:0'lambda_5/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskО
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╜
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3lambda_5/strided_slice:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( Р
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╤
conv2d_15/Conv2DConv2D*batch_normalization_5/FusedBatchNormV3:y:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Ж
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ы
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:         KK о
max_pooling2d_15/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
С
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0╔
conv2d_16/Conv2DConv2D!max_pooling2d_15/MaxPool:output:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
З
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аm
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ап
max_pooling2d_16/MaxPoolMaxPoolconv2d_16/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
Т
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╔
conv2d_17/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
З
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аm
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:         Ап
max_pooling2d_17/MaxPoolMaxPoolconv2d_17/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
}
dropout_15/IdentityIdentity!max_pooling2d_17/MaxPool:output:0*
T0*0
_output_shapes
:         		А`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  И
flatten_5/ReshapeReshapedropout_15/Identity:output:0flatten_5/Const:output:0*
T0*)
_output_shapes
:         АвЙ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Р
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:         Аo
dropout_16/IdentityIdentitydense_15/Relu:activations:0*
T0*(
_output_shapes
:         АИ
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
dense_16/MatMulMatMuldropout_16/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЕ
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Т
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аc
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:         Аo
dropout_17/IdentityIdentitydense_16/Relu:activations:0*
T0*(
_output_shapes
:         Аг
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ь
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0И
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ы
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0И
"dense_16/kernel/Regularizer/L2LossL2Loss9dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0+dense_16/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: l
IdentityIdentitydropout_17/Identity:output:0^NoOp*
T0*(
_output_shapes
:         А·
NoOpNoOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp1dense_16/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Щ
e
,__inference_dropout_15_layer_call_fn_1383216

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
G__inference_dropout_15_layer_call_and_return_conditional_losses_1381726x
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
Т
┤
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1381476

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp|
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
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0К
#conv2d_15/kernel/Regularizer/L2LossL2Loss:conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<г
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0,conv2d_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         KK м
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp2conv2d_15/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╠
b
F__inference_flatten_5_layer_call_and_return_conditional_losses_1381532

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
╠

f
G__inference_dropout_15_layer_call_and_return_conditional_losses_1383233

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
%__inference_CNN_layer_call_fn_1382152
%__inference_CNN_layer_call_fn_1382513
%__inference_CNN_layer_call_fn_1382550
%__inference_CNN_layer_call_fn_1382323│
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
@__inference_CNN_layer_call_and_return_conditional_losses_1382634
@__inference_CNN_layer_call_and_return_conditional_losses_1382739
@__inference_CNN_layer_call_and_return_conditional_losses_1382373
@__inference_CNN_layer_call_and_return_conditional_losses_1382423│
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
"__inference__wrapped_model_1381335input_1"Ш
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
__inference_call_1320321
__inference_call_1320393
__inference_call_1320465│
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
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
*:( 2conv2d_15/kernel
: 2conv2d_15/bias
+:) А2conv2d_16/kernel
:А2conv2d_16/bias
,:*АА2conv2d_17/kernel
:А2conv2d_17/bias
$:"АвА2dense_15/kernel
:А2dense_15/bias
#:!
АА2dense_16/kernel
:А2dense_16/bias
": 	А2dense_17/kernel
:2dense_17/bias
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
%__inference_CNN_layer_call_fn_1382152input_1"│
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
%__inference_CNN_layer_call_fn_1382513inputs"│
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
%__inference_CNN_layer_call_fn_1382550inputs"│
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
%__inference_CNN_layer_call_fn_1382323input_1"│
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
@__inference_CNN_layer_call_and_return_conditional_losses_1382634inputs"│
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
@__inference_CNN_layer_call_and_return_conditional_losses_1382739inputs"│
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
@__inference_CNN_layer_call_and_return_conditional_losses_1382373input_1"│
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
@__inference_CNN_layer_call_and_return_conditional_losses_1382423input_1"│
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
.__inference_sequential_5_layer_call_fn_1381634
.__inference_sequential_5_layer_call_fn_1382796
.__inference_sequential_5_layer_call_fn_1382829
.__inference_sequential_5_layer_call_fn_1381935┐
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1382906
I__inference_sequential_5_layer_call_and_return_conditional_losses_1383004
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381993
I__inference_sequential_5_layer_call_and_return_conditional_losses_1382051┐
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
*__inference_dense_17_layer_call_fn_1383013в
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
E__inference_dense_17_layer_call_and_return_conditional_losses_1383024в
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
__inference_call_1320321inputs"│
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
__inference_call_1320393inputs"│
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
__inference_call_1320465inputs"│
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
%__inference_signature_wrapper_1382476input_1"Ф
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
*__inference_lambda_5_layer_call_fn_1383029
*__inference_lambda_5_layer_call_fn_1383034┐
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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1383042
E__inference_lambda_5_layer_call_and_return_conditional_losses_1383050┐
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
7__inference_batch_normalization_5_layer_call_fn_1383063
7__inference_batch_normalization_5_layer_call_fn_1383076│
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1383094
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1383112│
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
+__inference_conv2d_15_layer_call_fn_1383121в
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
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1383136в
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
2__inference_max_pooling2d_15_layer_call_fn_1383141в
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
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1383146в
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
+__inference_conv2d_16_layer_call_fn_1383155в
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
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1383166в
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
2__inference_max_pooling2d_16_layer_call_fn_1383171в
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
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1383176в
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
+__inference_conv2d_17_layer_call_fn_1383185в
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
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1383196в
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
2__inference_max_pooling2d_17_layer_call_fn_1383201в
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
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1383206в
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
,__inference_dropout_15_layer_call_fn_1383211
,__inference_dropout_15_layer_call_fn_1383216│
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
G__inference_dropout_15_layer_call_and_return_conditional_losses_1383221
G__inference_dropout_15_layer_call_and_return_conditional_losses_1383233│
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
+__inference_flatten_5_layer_call_fn_1383238в
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
F__inference_flatten_5_layer_call_and_return_conditional_losses_1383244в
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
*__inference_dense_15_layer_call_fn_1383253в
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
E__inference_dense_15_layer_call_and_return_conditional_losses_1383268в
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
,__inference_dropout_16_layer_call_fn_1383273
,__inference_dropout_16_layer_call_fn_1383278│
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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1383283
G__inference_dropout_16_layer_call_and_return_conditional_losses_1383295│
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
*__inference_dense_16_layer_call_fn_1383304в
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
E__inference_dense_16_layer_call_and_return_conditional_losses_1383319в
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
,__inference_dropout_17_layer_call_fn_1383324
,__inference_dropout_17_layer_call_fn_1383329│
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
G__inference_dropout_17_layer_call_and_return_conditional_losses_1383334
G__inference_dropout_17_layer_call_and_return_conditional_losses_1383346│
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
__inference_loss_fn_0_1383355П
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
__inference_loss_fn_1_1383364П
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
__inference_loss_fn_2_1383373П
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
.__inference_sequential_5_layer_call_fn_1381634lambda_5_input"┐
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
.__inference_sequential_5_layer_call_fn_1382796inputs"┐
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
.__inference_sequential_5_layer_call_fn_1382829inputs"┐
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
.__inference_sequential_5_layer_call_fn_1381935lambda_5_input"┐
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1382906inputs"┐
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1383004inputs"┐
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381993lambda_5_input"┐
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1382051lambda_5_input"┐
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
*__inference_dense_17_layer_call_fn_1383013inputs"в
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
E__inference_dense_17_layer_call_and_return_conditional_losses_1383024inputs"в
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
.:,2"Adam/m/batch_normalization_5/gamma
.:,2"Adam/v/batch_normalization_5/gamma
-:+2!Adam/m/batch_normalization_5/beta
-:+2!Adam/v/batch_normalization_5/beta
/:- 2Adam/m/conv2d_15/kernel
/:- 2Adam/v/conv2d_15/kernel
!: 2Adam/m/conv2d_15/bias
!: 2Adam/v/conv2d_15/bias
0:. А2Adam/m/conv2d_16/kernel
0:. А2Adam/v/conv2d_16/kernel
": А2Adam/m/conv2d_16/bias
": А2Adam/v/conv2d_16/bias
1:/АА2Adam/m/conv2d_17/kernel
1:/АА2Adam/v/conv2d_17/kernel
": А2Adam/m/conv2d_17/bias
": А2Adam/v/conv2d_17/bias
):'АвА2Adam/m/dense_15/kernel
):'АвА2Adam/v/dense_15/kernel
!:А2Adam/m/dense_15/bias
!:А2Adam/v/dense_15/bias
(:&
АА2Adam/m/dense_16/kernel
(:&
АА2Adam/v/dense_16/kernel
!:А2Adam/m/dense_16/bias
!:А2Adam/v/dense_16/bias
':%	А2Adam/m/dense_17/kernel
':%	А2Adam/v/dense_17/kernel
 :2Adam/m/dense_17/bias
 :2Adam/v/dense_17/bias
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
*__inference_lambda_5_layer_call_fn_1383029inputs"┐
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
*__inference_lambda_5_layer_call_fn_1383034inputs"┐
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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1383042inputs"┐
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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1383050inputs"┐
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
7__inference_batch_normalization_5_layer_call_fn_1383063inputs"│
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
7__inference_batch_normalization_5_layer_call_fn_1383076inputs"│
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1383094inputs"│
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1383112inputs"│
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
+__inference_conv2d_15_layer_call_fn_1383121inputs"в
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
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1383136inputs"в
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
2__inference_max_pooling2d_15_layer_call_fn_1383141inputs"в
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
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1383146inputs"в
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
+__inference_conv2d_16_layer_call_fn_1383155inputs"в
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
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1383166inputs"в
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
2__inference_max_pooling2d_16_layer_call_fn_1383171inputs"в
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
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1383176inputs"в
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
+__inference_conv2d_17_layer_call_fn_1383185inputs"в
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
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1383196inputs"в
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
2__inference_max_pooling2d_17_layer_call_fn_1383201inputs"в
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
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1383206inputs"в
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
,__inference_dropout_15_layer_call_fn_1383211inputs"│
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
,__inference_dropout_15_layer_call_fn_1383216inputs"│
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
G__inference_dropout_15_layer_call_and_return_conditional_losses_1383221inputs"│
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
G__inference_dropout_15_layer_call_and_return_conditional_losses_1383233inputs"│
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
+__inference_flatten_5_layer_call_fn_1383238inputs"в
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
F__inference_flatten_5_layer_call_and_return_conditional_losses_1383244inputs"в
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
*__inference_dense_15_layer_call_fn_1383253inputs"в
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
E__inference_dense_15_layer_call_and_return_conditional_losses_1383268inputs"в
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
,__inference_dropout_16_layer_call_fn_1383273inputs"│
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
,__inference_dropout_16_layer_call_fn_1383278inputs"│
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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1383283inputs"│
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
G__inference_dropout_16_layer_call_and_return_conditional_losses_1383295inputs"│
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
*__inference_dense_16_layer_call_fn_1383304inputs"в
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
E__inference_dense_16_layer_call_and_return_conditional_losses_1383319inputs"в
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
,__inference_dropout_17_layer_call_fn_1383324inputs"│
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
,__inference_dropout_17_layer_call_fn_1383329inputs"│
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
G__inference_dropout_17_layer_call_and_return_conditional_losses_1383334inputs"│
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
G__inference_dropout_17_layer_call_and_return_conditional_losses_1383346inputs"│
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
__inference_loss_fn_0_1383355"П
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
__inference_loss_fn_1_1383364"П
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
__inference_loss_fn_2_1383373"П
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
@__inference_CNN_layer_call_and_return_conditional_losses_1382373~<в9
2в/
)К&
input_1         KK
p 
к ",в)
"К
tensor_0         
Ъ ┬
@__inference_CNN_layer_call_and_return_conditional_losses_1382423~<в9
2в/
)К&
input_1         KK
p
к ",в)
"К
tensor_0         
Ъ ┴
@__inference_CNN_layer_call_and_return_conditional_losses_1382634};в8
1в.
(К%
inputs         KK
p 
к ",в)
"К
tensor_0         
Ъ ┴
@__inference_CNN_layer_call_and_return_conditional_losses_1382739};в8
1в.
(К%
inputs         KK
p
к ",в)
"К
tensor_0         
Ъ Ь
%__inference_CNN_layer_call_fn_1382152s<в9
2в/
)К&
input_1         KK
p 
к "!К
unknown         Ь
%__inference_CNN_layer_call_fn_1382323s<в9
2в/
)К&
input_1         KK
p
к "!К
unknown         Ы
%__inference_CNN_layer_call_fn_1382513r;в8
1в.
(К%
inputs         KK
p 
к "!К
unknown         Ы
%__inference_CNN_layer_call_fn_1382550r;в8
1в.
(К%
inputs         KK
p
к "!К
unknown         и
"__inference__wrapped_model_1381335Б8в5
.в+
)К&
input_1         KK
к "3к0
.
output_1"К
output_1         Ї
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1383094ЭMвJ
Cв@
:К7
inputs+                           
p 
к "FвC
<К9
tensor_0+                           
Ъ Ї
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1383112ЭMвJ
Cв@
:К7
inputs+                           
p
к "FвC
<К9
tensor_0+                           
Ъ ╬
7__inference_batch_normalization_5_layer_call_fn_1383063ТMвJ
Cв@
:К7
inputs+                           
p 
к ";К8
unknown+                           ╬
7__inference_batch_normalization_5_layer_call_fn_1383076ТMвJ
Cв@
:К7
inputs+                           
p
к ";К8
unknown+                           ~
__inference_call_1320321b3в0
)в&
 К
inputsАKK
p
к "К
unknown	А~
__inference_call_1320393b3в0
)в&
 К
inputsАKK
p 
к "К
unknown	АО
__inference_call_1320465r;в8
1в.
(К%
inputs         KK
p 
к "!К
unknown         ╜
F__inference_conv2d_15_layer_call_and_return_conditional_losses_1383136s7в4
-в*
(К%
inputs         KK
к "4в1
*К'
tensor_0         KK 
Ъ Ч
+__inference_conv2d_15_layer_call_fn_1383121h7в4
-в*
(К%
inputs         KK
к ")К&
unknown         KK ╛
F__inference_conv2d_16_layer_call_and_return_conditional_losses_1383166t7в4
-в*
(К%
inputs         %% 
к "5в2
+К(
tensor_0         %%А
Ъ Ш
+__inference_conv2d_16_layer_call_fn_1383155i7в4
-в*
(К%
inputs         %% 
к "*К'
unknown         %%А┐
F__inference_conv2d_17_layer_call_and_return_conditional_losses_1383196u8в5
.в+
)К&
inputs         А
к "5в2
+К(
tensor_0         А
Ъ Щ
+__inference_conv2d_17_layer_call_fn_1383185j8в5
.в+
)К&
inputs         А
к "*К'
unknown         Ап
E__inference_dense_15_layer_call_and_return_conditional_losses_1383268f1в.
'в$
"К
inputs         Ав
к "-в*
#К 
tensor_0         А
Ъ Й
*__inference_dense_15_layer_call_fn_1383253[1в.
'в$
"К
inputs         Ав
к ""К
unknown         Ао
E__inference_dense_16_layer_call_and_return_conditional_losses_1383319e0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ И
*__inference_dense_16_layer_call_fn_1383304Z0в-
&в#
!К
inputs         А
к ""К
unknown         Ан
E__inference_dense_17_layer_call_and_return_conditional_losses_1383024d0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ З
*__inference_dense_17_layer_call_fn_1383013Y0в-
&в#
!К
inputs         А
к "!К
unknown         └
G__inference_dropout_15_layer_call_and_return_conditional_losses_1383221u<в9
2в/
)К&
inputs         		А
p 
к "5в2
+К(
tensor_0         		А
Ъ └
G__inference_dropout_15_layer_call_and_return_conditional_losses_1383233u<в9
2в/
)К&
inputs         		А
p
к "5в2
+К(
tensor_0         		А
Ъ Ъ
,__inference_dropout_15_layer_call_fn_1383211j<в9
2в/
)К&
inputs         		А
p 
к "*К'
unknown         		АЪ
,__inference_dropout_15_layer_call_fn_1383216j<в9
2в/
)К&
inputs         		А
p
к "*К'
unknown         		А░
G__inference_dropout_16_layer_call_and_return_conditional_losses_1383283e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ ░
G__inference_dropout_16_layer_call_and_return_conditional_losses_1383295e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ К
,__inference_dropout_16_layer_call_fn_1383273Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         АК
,__inference_dropout_16_layer_call_fn_1383278Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         А░
G__inference_dropout_17_layer_call_and_return_conditional_losses_1383334e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ ░
G__inference_dropout_17_layer_call_and_return_conditional_losses_1383346e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ К
,__inference_dropout_17_layer_call_fn_1383324Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         АК
,__inference_dropout_17_layer_call_fn_1383329Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         А┤
F__inference_flatten_5_layer_call_and_return_conditional_losses_1383244j8в5
.в+
)К&
inputs         		А
к ".в+
$К!
tensor_0         Ав
Ъ О
+__inference_flatten_5_layer_call_fn_1383238_8в5
.в+
)К&
inputs         		А
к "#К 
unknown         Ав└
E__inference_lambda_5_layer_call_and_return_conditional_losses_1383042w?в<
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
E__inference_lambda_5_layer_call_and_return_conditional_losses_1383050w?в<
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
*__inference_lambda_5_layer_call_fn_1383029l?в<
5в2
(К%
inputs         KK

 
p 
к ")К&
unknown         KKЪ
*__inference_lambda_5_layer_call_fn_1383034l?в<
5в2
(К%
inputs         KK

 
p
к ")К&
unknown         KKE
__inference_loss_fn_0_1383355$в

в 
к "К
unknown E
__inference_loss_fn_1_1383364$в

в 
к "К
unknown E
__inference_loss_fn_2_1383373$в

в 
к "К
unknown ў
M__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_1383146еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_15_layer_call_fn_1383141ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ў
M__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_1383176еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_16_layer_call_fn_1383171ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ў
M__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_1383206еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_17_layer_call_fn_1383201ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ╓
I__inference_sequential_5_layer_call_and_return_conditional_losses_1381993ИGвD
=в:
0К-
lambda_5_input         KK
p 

 
к "-в*
#К 
tensor_0         А
Ъ ╓
I__inference_sequential_5_layer_call_and_return_conditional_losses_1382051ИGвD
=в:
0К-
lambda_5_input         KK
p

 
к "-в*
#К 
tensor_0         А
Ъ ╬
I__inference_sequential_5_layer_call_and_return_conditional_losses_1382906А?в<
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
I__inference_sequential_5_layer_call_and_return_conditional_losses_1383004А?в<
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
.__inference_sequential_5_layer_call_fn_1381634}GвD
=в:
0К-
lambda_5_input         KK
p 

 
к ""К
unknown         Ап
.__inference_sequential_5_layer_call_fn_1381935}GвD
=в:
0К-
lambda_5_input         KK
p

 
к ""К
unknown         Аз
.__inference_sequential_5_layer_call_fn_1382796u?в<
5в2
(К%
inputs         KK
p 

 
к ""К
unknown         Аз
.__inference_sequential_5_layer_call_fn_1382829u?в<
5в2
(К%
inputs         KK
p

 
к ""К
unknown         А╢
%__inference_signature_wrapper_1382476МCв@
в 
9к6
4
input_1)К&
input_1         KK"3к0
.
output_1"К
output_1         