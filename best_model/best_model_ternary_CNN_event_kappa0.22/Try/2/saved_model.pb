ВУ
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
 И"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58щ╖
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
~
Adam/v/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_8/bias
w
'Adam/v/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_8/bias
w
'Adam/m/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/bias*
_output_shapes
:*
dtype0
З
Adam/v/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/v/dense_8/kernel
А
)Adam/v/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/kernel*
_output_shapes
:	А*
dtype0
З
Adam/m/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/m/dense_8/kernel
А
)Adam/m/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/kernel*
_output_shapes
:	А*
dtype0

Adam/v/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/v/dense_7/bias
x
'Adam/v/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/bias*
_output_shapes	
:А*
dtype0

Adam/m/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/m/dense_7/bias
x
'Adam/m/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/bias*
_output_shapes	
:А*
dtype0
И
Adam/v/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/v/dense_7/kernel
Б
)Adam/v/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/kernel* 
_output_shapes
:
АА*
dtype0
И
Adam/m/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_nameAdam/m/dense_7/kernel
Б
)Adam/m/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/kernel* 
_output_shapes
:
АА*
dtype0

Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/v/dense_6/bias
x
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes	
:А*
dtype0

Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/m/dense_6/bias
x
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes	
:А*
dtype0
Й
Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*&
shared_nameAdam/v/dense_6/kernel
В
)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel*!
_output_shapes
:АвА*
dtype0
Й
Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*&
shared_nameAdam/m/dense_6/kernel
В
)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel*!
_output_shapes
:АвА*
dtype0
Б
Adam/v/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/conv2d_8/bias
z
(Adam/v/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/conv2d_8/bias
z
(Adam/m/conv2d_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/bias*
_output_shapes	
:А*
dtype0
Т
Adam/v/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/v/conv2d_8/kernel
Л
*Adam/v/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_8/kernel*(
_output_shapes
:АА*
dtype0
Т
Adam/m/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/m/conv2d_8/kernel
Л
*Adam/m/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_8/kernel*(
_output_shapes
:АА*
dtype0
Б
Adam/v/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/conv2d_7/bias
z
(Adam/v/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/conv2d_7/bias
z
(Adam/m/conv2d_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/bias*
_output_shapes	
:А*
dtype0
С
Adam/v/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*'
shared_nameAdam/v/conv2d_7/kernel
К
*Adam/v/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_7/kernel*'
_output_shapes
: А*
dtype0
С
Adam/m/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*'
shared_nameAdam/m/conv2d_7/kernel
К
*Adam/m/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_7/kernel*'
_output_shapes
: А*
dtype0
А
Adam/v/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/conv2d_6/bias
y
(Adam/v/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/bias*
_output_shapes
: *
dtype0
А
Adam/m/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/conv2d_6/bias
y
(Adam/m/conv2d_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/bias*
_output_shapes
: *
dtype0
Р
Adam/v/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/v/conv2d_6/kernel
Й
*Adam/v/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_6/kernel*&
_output_shapes
: *
dtype0
Р
Adam/m/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/m/conv2d_6/kernel
Й
*Adam/m/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_6/kernel*&
_output_shapes
: *
dtype0
Ъ
!Adam/v/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/batch_normalization_2/beta
У
5Adam/v/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_2/beta*
_output_shapes
:*
dtype0
Ъ
!Adam/m/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/batch_normalization_2/beta
У
5Adam/m/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_2/beta*
_output_shapes
:*
dtype0
Ь
"Adam/v/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_2/gamma
Х
6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_2/gamma*
_output_shapes
:*
dtype0
Ь
"Adam/m/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_2/gamma
Х
6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_2/gamma*
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
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	А*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:А*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:А*
dtype0
{
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АвА*
shared_namedense_6/kernel
t
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*!
_output_shapes
:АвА*
dtype0
s
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_8/bias
l
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes	
:А*
dtype0
Д
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv2d_8/kernel
}
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*(
_output_shapes
:АА*
dtype0
s
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_7/bias
l
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes	
:А*
dtype0
Г
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А* 
shared_nameconv2d_7/kernel
|
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*'
_output_shapes
: А*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
: *
dtype0
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: *
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:*
dtype0
К
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
П
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias*
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
GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_597720

NoOpNoOp
▄Б
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЦБ
valueЛБBЗБ B А
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
VARIABLE_VALUEbatch_normalization_2/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_2/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!batch_normalization_2/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%batch_normalization_2/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_6/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_6/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_7/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_7/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_8/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_8/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_6/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_6/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_7/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_7/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_8/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_8/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE"Adam/m/batch_normalization_2/gamma1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/batch_normalization_2/gamma1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/m/batch_normalization_2/beta1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/v/batch_normalization_2/beta1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_6/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_6/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_6/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_6/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_7/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_7/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_7/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_7/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_8/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_8/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_8/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_8/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_6/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_6/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_6/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_6/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_7/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_7/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_7/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_7/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_8/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_8/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_8/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_8/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
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
╓
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_2/beta/Read/ReadVariableOp5Adam/v/batch_normalization_2/beta/Read/ReadVariableOp*Adam/m/conv2d_6/kernel/Read/ReadVariableOp*Adam/v/conv2d_6/kernel/Read/ReadVariableOp(Adam/m/conv2d_6/bias/Read/ReadVariableOp(Adam/v/conv2d_6/bias/Read/ReadVariableOp*Adam/m/conv2d_7/kernel/Read/ReadVariableOp*Adam/v/conv2d_7/kernel/Read/ReadVariableOp(Adam/m/conv2d_7/bias/Read/ReadVariableOp(Adam/v/conv2d_7/bias/Read/ReadVariableOp*Adam/m/conv2d_8/kernel/Read/ReadVariableOp*Adam/v/conv2d_8/kernel/Read/ReadVariableOp(Adam/m/conv2d_8/bias/Read/ReadVariableOp(Adam/v/conv2d_8/bias/Read/ReadVariableOp)Adam/m/dense_6/kernel/Read/ReadVariableOp)Adam/v/dense_6/kernel/Read/ReadVariableOp'Adam/m/dense_6/bias/Read/ReadVariableOp'Adam/v/dense_6/bias/Read/ReadVariableOp)Adam/m/dense_7/kernel/Read/ReadVariableOp)Adam/v/dense_7/kernel/Read/ReadVariableOp'Adam/m/dense_7/bias/Read/ReadVariableOp'Adam/v/dense_7/bias/Read/ReadVariableOp)Adam/m/dense_8/kernel/Read/ReadVariableOp)Adam/v/dense_8/kernel/Read/ReadVariableOp'Adam/m/dense_8/bias/Read/ReadVariableOp'Adam/v/dense_8/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*?
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
GPU 2J 8В *(
f#R!
__inference__traced_save_598790
щ

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias	iterationlearning_rate"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/conv2d_6/kernelAdam/v/conv2d_6/kernelAdam/m/conv2d_6/biasAdam/v/conv2d_6/biasAdam/m/conv2d_7/kernelAdam/v/conv2d_7/kernelAdam/m/conv2d_7/biasAdam/v/conv2d_7/biasAdam/m/conv2d_8/kernelAdam/v/conv2d_8/kernelAdam/m/conv2d_8/biasAdam/v/conv2d_8/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biastotal_1count_1totalcount*>
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
GPU 2J 8В *+
f&R$
"__inference__traced_restore_598950Он
╜
E
)__inference_lambda_2_layer_call_fn_598273

inputs
identity╖
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
GPU 2J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_596694h
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
▄
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_596832

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
Ў
¤
-__inference_sequential_2_layer_call_fn_598040

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
identityИвStatefulPartitionedCall¤
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_596847p
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
Ь
л
C__inference_dense_6_layer_call_and_return_conditional_losses_596793

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpw
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
:         АТ
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ак
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
┼
╝
!__inference__wrapped_model_596579
input_1

cnn_596545:

cnn_596547:

cnn_596549:

cnn_596551:$

cnn_596553: 

cnn_596555: %

cnn_596557: А

cnn_596559:	А&

cnn_596561:АА

cnn_596563:	А

cnn_596565:АвА

cnn_596567:	А

cnn_596569:
АА

cnn_596571:	А

cnn_596573:	А

cnn_596575:
identityИвCNN/StatefulPartitionedCall∙
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1
cnn_596545
cnn_596547
cnn_596549
cnn_596551
cnn_596553
cnn_596555
cnn_596557
cnn_596559
cnn_596561
cnn_596563
cnn_596565
cnn_596567
cnn_596569
cnn_596571
cnn_596573
cnn_596575*
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
GPU 2J 8В * 
fR
__inference_call_529376s
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
Ч
к
C__inference_dense_7_layer_call_and_return_conditional_losses_596821

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpv
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
:         АС
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ак
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_598390

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
Ш	
▓
__inference_loss_fn_1_598608N
9dense_6_kernel_regularizer_l2loss_readvariableop_resource:АвА
identityИв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpн
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_6_kernel_regularizer_l2loss_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_6/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp
╩

d
E__inference_dropout_6_layer_call_and_return_conditional_losses_596970

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
Т

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_598590

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
П
А
D__inference_conv2d_8_layer_call_and_return_conditional_losses_596756

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
ъ
Ю
)__inference_conv2d_6_layer_call_fn_598365

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallс
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_596720w
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
є╨
с
"__inference__traced_restore_598950
file_prefix:
,assignvariableop_batch_normalization_2_gamma:;
-assignvariableop_1_batch_normalization_2_beta:B
4assignvariableop_2_batch_normalization_2_moving_mean:F
8assignvariableop_3_batch_normalization_2_moving_variance:<
"assignvariableop_4_conv2d_6_kernel: .
 assignvariableop_5_conv2d_6_bias: =
"assignvariableop_6_conv2d_7_kernel: А/
 assignvariableop_7_conv2d_7_bias:	А>
"assignvariableop_8_conv2d_8_kernel:АА/
 assignvariableop_9_conv2d_8_bias:	А7
"assignvariableop_10_dense_6_kernel:АвА/
 assignvariableop_11_dense_6_bias:	А6
"assignvariableop_12_dense_7_kernel:
АА/
 assignvariableop_13_dense_7_bias:	А5
"assignvariableop_14_dense_8_kernel:	А.
 assignvariableop_15_dense_8_bias:'
assignvariableop_16_iteration:	 +
!assignvariableop_17_learning_rate: D
6assignvariableop_18_adam_m_batch_normalization_2_gamma:D
6assignvariableop_19_adam_v_batch_normalization_2_gamma:C
5assignvariableop_20_adam_m_batch_normalization_2_beta:C
5assignvariableop_21_adam_v_batch_normalization_2_beta:D
*assignvariableop_22_adam_m_conv2d_6_kernel: D
*assignvariableop_23_adam_v_conv2d_6_kernel: 6
(assignvariableop_24_adam_m_conv2d_6_bias: 6
(assignvariableop_25_adam_v_conv2d_6_bias: E
*assignvariableop_26_adam_m_conv2d_7_kernel: АE
*assignvariableop_27_adam_v_conv2d_7_kernel: А7
(assignvariableop_28_adam_m_conv2d_7_bias:	А7
(assignvariableop_29_adam_v_conv2d_7_bias:	АF
*assignvariableop_30_adam_m_conv2d_8_kernel:ААF
*assignvariableop_31_adam_v_conv2d_8_kernel:АА7
(assignvariableop_32_adam_m_conv2d_8_bias:	А7
(assignvariableop_33_adam_v_conv2d_8_bias:	А>
)assignvariableop_34_adam_m_dense_6_kernel:АвА>
)assignvariableop_35_adam_v_dense_6_kernel:АвА6
'assignvariableop_36_adam_m_dense_6_bias:	А6
'assignvariableop_37_adam_v_dense_6_bias:	А=
)assignvariableop_38_adam_m_dense_7_kernel:
АА=
)assignvariableop_39_adam_v_dense_7_kernel:
АА6
'assignvariableop_40_adam_m_dense_7_bias:	А6
'assignvariableop_41_adam_v_dense_7_bias:	А<
)assignvariableop_42_adam_m_dense_8_kernel:	А<
)assignvariableop_43_adam_v_dense_8_kernel:	А5
'assignvariableop_44_adam_m_dense_8_bias:5
'assignvariableop_45_adam_v_dense_8_bias:%
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
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_2_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_2_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_2_moving_meanIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_2_moving_varianceIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_6_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_6_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_7_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_7_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_8_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_8_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_6_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_6_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_7_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_7_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_8_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_8_biasIdentity_15:output:0"/device:CPU:0*&
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
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_m_batch_normalization_2_gammaIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╧
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_v_batch_normalization_2_gammaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_m_batch_normalization_2_betaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_v_batch_normalization_2_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_m_conv2d_6_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_v_conv2d_6_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_m_conv2d_6_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_v_conv2d_6_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_m_conv2d_7_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_v_conv2d_7_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_m_conv2d_7_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_v_conv2d_7_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_m_conv2d_8_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_v_conv2d_8_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_m_conv2d_8_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_v_conv2d_8_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_m_dense_6_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_v_dense_6_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_m_dense_6_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_v_dense_6_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_m_dense_7_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_v_dense_7_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_m_dense_7_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_v_dense_7_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_m_dense_8_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_v_dense_8_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_m_dense_8_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_v_dense_8_biasIdentity_45:output:0"/device:CPU:0*&
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
▄
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_598578

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
├
F
*__inference_dropout_6_layer_call_fn_598455

inputs
identity╣
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
GPU 2J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_596768i
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
╣
н
$__inference_CNN_layer_call_fn_597794

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
identityИвStatefulPartitionedCallН
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
GPU 2J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_597495o
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
У
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_596664

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
С	
╤
6__inference_batch_normalization_2_layer_call_fn_598307

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЪ
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
GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_596601Й
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
Нy
Э
?__inference_CNN_layer_call_and_return_conditional_losses_597878

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АH
3sequential_2_dense_6_matmul_readvariableop_resource:АвАC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpвBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в1sequential_2/batch_normalization_2/ReadVariableOpв3sequential_2/batch_normalization_2/ReadVariableOp_1в,sequential_2/conv2d_6/BiasAdd/ReadVariableOpв+sequential_2/conv2d_6/Conv2D/ReadVariableOpв,sequential_2/conv2d_7/BiasAdd/ReadVariableOpв+sequential_2/conv2d_7/Conv2D/ReadVariableOpв,sequential_2/conv2d_8/BiasAdd/ReadVariableOpв+sequential_2/conv2d_8/Conv2D/ReadVariableOpв+sequential_2/dense_6/BiasAdd/ReadVariableOpв*sequential_2/dense_6/MatMul/ReadVariableOpв+sequential_2/dense_7/BiasAdd/ReadVariableOpв*sequential_2/dense_7/MatMul/ReadVariableOpВ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( и
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Ю
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Д
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╞
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
й
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0э
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Я
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0└
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЕ
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╟
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
к
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0э
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Я
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0└
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЕ
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:         А╟
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Х
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:         		Аm
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  о
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*)
_output_shapes
:         Авб
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╡
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         АЗ
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0*(
_output_shapes
:         Аа
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╢
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:         АЗ
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0*(
_output_shapes
:         АЕ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ы
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         о
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: з
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ж
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ф
NoOpNoOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_598450

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
Е
▒
D__inference_conv2d_6_layer_call_and_return_conditional_losses_596720

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp|
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
:         KK Ш
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         KK л
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
У
g
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_596676

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
╕
L
0__inference_max_pooling2d_6_layer_call_fn_598385

inputs
identity┘
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_596652Г
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
Й%
д
?__inference_CNN_layer_call_and_return_conditional_losses_597495

inputs!
sequential_2_597448:!
sequential_2_597450:!
sequential_2_597452:!
sequential_2_597454:-
sequential_2_597456: !
sequential_2_597458: .
sequential_2_597460: А"
sequential_2_597462:	А/
sequential_2_597464:АА"
sequential_2_597466:	А(
sequential_2_597468:АвА"
sequential_2_597470:	А'
sequential_2_597472:
АА"
sequential_2_597474:	А!
dense_8_597477:	А
dense_8_597479:
identityИв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpвdense_8/StatefulPartitionedCallв$sequential_2/StatefulPartitionedCallУ
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_597448sequential_2_597450sequential_2_597452sequential_2_597454sequential_2_597456sequential_2_597458sequential_2_597460sequential_2_597462sequential_2_597464sequential_2_597466sequential_2_597468sequential_2_597470sequential_2_597472sequential_2_597474*
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_597115У
dense_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0dense_8_597477dense_8_597479*
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
GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_597342Н
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597456*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: З
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597468*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597472* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_8/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ъ~
╫
H__inference_sequential_2_layer_call_and_return_conditional_losses_598248

inputs;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: B
'conv2d_7_conv2d_readvariableop_resource: А7
(conv2d_7_biasadd_readvariableop_resource:	АC
'conv2d_8_conv2d_readvariableop_resource:АА7
(conv2d_8_biasadd_readvariableop_resource:	А;
&dense_6_matmul_readvariableop_resource:АвА6
'dense_6_biasadd_readvariableop_resource:	А:
&dense_7_matmul_readvariableop_resource:
АА6
'dense_7_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_2/AssignNewValueв&batch_normalization_2/AssignNewValue_1в5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1вconv2d_6/BiasAdd/ReadVariableOpвconv2d_6/Conv2D/ReadVariableOpв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpвconv2d_7/BiasAdd/ReadVariableOpвconv2d_7/Conv2D/ReadVariableOpвconv2d_8/BiasAdd/ReadVariableOpвconv2d_8/Conv2D/ReadVariableOpвdense_6/BiasAdd/ReadVariableOpвdense_6/MatMul/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpвdense_7/BiasAdd/ReadVariableOpвdense_7/MatMul/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpu
lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Р
lambda_2/strided_sliceStridedSliceinputs%lambda_2/strided_slice/stack:output:0'lambda_2/strided_slice/stack_1:output:0'lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskО
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╦
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3lambda_2/strided_slice:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<Ю
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(и
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(О
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╧
conv2d_6/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         KK м
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
П
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0╞
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Е
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аk
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ан
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
Р
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╞
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Е
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аk
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:         Ан
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?Ы
dropout_6/dropout/MulMul max_pooling2d_8/MaxPool:output:0 dropout_6/dropout/Const:output:0*
T0*0
_output_shapes
:         		Аg
dropout_6/dropout/ShapeShape max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:й
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=═
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		А^
dropout_6/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ─
dropout_6/dropout/SelectV2SelectV2"dropout_6/dropout/GreaterEqual:z:0dropout_6/dropout/Mul:z:0"dropout_6/dropout/Const_1:output:0*
T0*0
_output_shapes
:         		А`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  П
flatten_2/ReshapeReshape#dropout_6/dropout/SelectV2:output:0flatten_2/Const:output:0*
T0*)
_output_shapes
:         АвЗ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0О
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         А\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Н
dropout_7/dropout/MulMuldense_6/Relu:activations:0 dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:         Аa
dropout_7/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:б
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?┼
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А^
dropout_7/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╝
dropout_7/dropout/SelectV2SelectV2"dropout_7/dropout/GreaterEqual:z:0dropout_7/dropout/Mul:z:0"dropout_7/dropout/Const_1:output:0*
T0*(
_output_shapes
:         АЖ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ч
dense_7/MatMulMatMul#dropout_7/dropout/SelectV2:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:         А\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Н
dropout_8/dropout/MulMuldense_7/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:         Аa
dropout_8/dropout/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:б
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?┼
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А^
dropout_8/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╝
dropout_8/dropout/SelectV2SelectV2"dropout_8/dropout/GreaterEqual:z:0dropout_8/dropout/Mul:z:0"dropout_8/dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аб
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ъ
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Щ
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity#dropout_8/dropout/SelectV2:output:0^NoOp*
T0*(
_output_shapes
:         А╜
NoOpNoOp%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
П	
╤
6__inference_batch_normalization_2_layer_call_fn_598320

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallШ
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
GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_596632Й
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
У
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_598420

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
╕
L
0__inference_max_pooling2d_7_layer_call_fn_598415

inputs
identity┘
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_596664Г
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
Т

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_596898

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
ЁЭ
З
?__inference_CNN_layer_call_and_return_conditional_losses_597983

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АH
3sequential_2_dense_6_matmul_readvariableop_resource:АвАC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpв1sequential_2/batch_normalization_2/AssignNewValueв3sequential_2/batch_normalization_2/AssignNewValue_1вBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в1sequential_2/batch_normalization_2/ReadVariableOpв3sequential_2/batch_normalization_2/ReadVariableOp_1в,sequential_2/conv2d_6/BiasAdd/ReadVariableOpв+sequential_2/conv2d_6/Conv2D/ReadVariableOpв,sequential_2/conv2d_7/BiasAdd/ReadVariableOpв+sequential_2/conv2d_7/Conv2D/ReadVariableOpв,sequential_2/conv2d_8/BiasAdd/ReadVariableOpв+sequential_2/conv2d_8/Conv2D/ReadVariableOpв+sequential_2/dense_6/BiasAdd/ReadVariableOpв*sequential_2/dense_6/MatMul/ReadVariableOpв+sequential_2/dense_7/BiasAdd/ReadVariableOpв*sequential_2/dense_7/MatMul/ReadVariableOpВ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Щ
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<╥
1sequential_2/batch_normalization_2/AssignNewValueAssignVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource@sequential_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0C^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(▄
3sequential_2/batch_normalization_2/AssignNewValue_1AssignVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceDsequential_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0E^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(и
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Ю
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Д
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╞
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
й
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0э
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Я
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0└
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЕ
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╟
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
к
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0э
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Я
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0└
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЕ
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:         А╟
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
i
$sequential_2/dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?┬
"sequential_2/dropout_6/dropout/MulMul-sequential_2/max_pooling2d_8/MaxPool:output:0-sequential_2/dropout_6/dropout/Const:output:0*
T0*0
_output_shapes
:         		АБ
$sequential_2/dropout_6/dropout/ShapeShape-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:├
;sequential_2/dropout_6/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_6/dropout/Shape:output:0*
T0*0
_output_shapes
:         		А*
dtype0r
-sequential_2/dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=Ї
+sequential_2/dropout_6/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_6/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_6/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		Аk
&sequential_2/dropout_6/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    °
'sequential_2/dropout_6/dropout/SelectV2SelectV2/sequential_2/dropout_6/dropout/GreaterEqual:z:0&sequential_2/dropout_6/dropout/Mul:z:0/sequential_2/dropout_6/dropout/Const_1:output:0*
T0*0
_output_shapes
:         		Аm
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  ╢
sequential_2/flatten_2/ReshapeReshape0sequential_2/dropout_6/dropout/SelectV2:output:0%sequential_2/flatten_2/Const:output:0*
T0*)
_output_shapes
:         Авб
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╡
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         Аi
$sequential_2/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @┤
"sequential_2/dropout_7/dropout/MulMul'sequential_2/dense_6/Relu:activations:0-sequential_2/dropout_7/dropout/Const:output:0*
T0*(
_output_shapes
:         А{
$sequential_2/dropout_7/dropout/ShapeShape'sequential_2/dense_6/Relu:activations:0*
T0*
_output_shapes
:╗
;sequential_2/dropout_7/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_7/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0r
-sequential_2/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ь
+sequential_2/dropout_7/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_7/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_7/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аk
&sequential_2/dropout_7/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ё
'sequential_2/dropout_7/dropout/SelectV2SelectV2/sequential_2/dropout_7/dropout/GreaterEqual:z:0&sequential_2/dropout_7/dropout/Mul:z:0/sequential_2/dropout_7/dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аа
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╛
sequential_2/dense_7/MatMulMatMul0sequential_2/dropout_7/dropout/SelectV2:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:         Аi
$sequential_2/dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @┤
"sequential_2/dropout_8/dropout/MulMul'sequential_2/dense_7/Relu:activations:0-sequential_2/dropout_8/dropout/Const:output:0*
T0*(
_output_shapes
:         А{
$sequential_2/dropout_8/dropout/ShapeShape'sequential_2/dense_7/Relu:activations:0*
T0*
_output_shapes
:╗
;sequential_2/dropout_8/dropout/random_uniform/RandomUniformRandomUniform-sequential_2/dropout_8/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0r
-sequential_2/dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ь
+sequential_2/dropout_8/dropout/GreaterEqualGreaterEqualDsequential_2/dropout_8/dropout/random_uniform/RandomUniform:output:06sequential_2/dropout_8/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аk
&sequential_2/dropout_8/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ё
'sequential_2/dropout_8/dropout/SelectV2SelectV2/sequential_2/dropout_8/dropout/GreaterEqual:z:0&sequential_2/dropout_8/dropout/Mul:z:0/sequential_2/dropout_8/dropout/Const_1:output:0*
T0*(
_output_shapes
:         АЕ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0г
dense_8/MatMulMatMul0sequential_2/dropout_8/dropout/SelectV2:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         о
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: з
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: ж
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╬
NoOpNoOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp2^sequential_2/batch_normalization_2/AssignNewValue4^sequential_2/batch_normalization_2/AssignNewValue_1C^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2f
1sequential_2/batch_normalization_2/AssignNewValue1sequential_2/batch_normalization_2/AssignNewValue2j
3sequential_2/batch_normalization_2/AssignNewValue_13sequential_2/batch_normalization_2/AssignNewValue_12И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┘
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_596694

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
╕
L
0__inference_max_pooling2d_8_layer_call_fn_598445

inputs
identity┘
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_596676Г
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
ё
б
)__inference_conv2d_8_layer_call_fn_598429

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallт
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_596756x
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
Ж
└
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_598356

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
ГK
°
H__inference_sequential_2_layer_call_and_return_conditional_losses_596847

inputs*
batch_normalization_2_596696:*
batch_normalization_2_596698:*
batch_normalization_2_596700:*
batch_normalization_2_596702:)
conv2d_6_596721: 
conv2d_6_596723: *
conv2d_7_596739: А
conv2d_7_596741:	А+
conv2d_8_596757:АА
conv2d_8_596759:	А#
dense_6_596794:АвА
dense_6_596796:	А"
dense_7_596822:
АА
dense_7_596824:	А
identityИв-batch_normalization_2/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв conv2d_7/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallвdense_6/StatefulPartitionedCallв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpвdense_7/StatefulPartitionedCallв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp└
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_596694З
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0batch_normalization_2_596696batch_normalization_2_596698batch_normalization_2_596700batch_normalization_2_596702*
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
GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_596601и
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_6_596721conv2d_6_596723*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_596720ё
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_596652Ы
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_596739conv2d_7_596741*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_596738Є
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_596664Ы
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_596757conv2d_8_596759*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_596756Є
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_596676х
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_596768╪
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_596776Й
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_596794dense_6_596796*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_596793▌
dropout_7/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_596804Й
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_7_596822dense_7_596824*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_596821▌
dropout_8/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_596832Й
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_596721*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_596794*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Б
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_7_596822* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity"dropout_8/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А╜
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_7/StatefulPartitionedCall1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Т

d
E__inference_dropout_7_layer_call_and_return_conditional_losses_598539

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
▄
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_596804

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
┴O
ф
H__inference_sequential_2_layer_call_and_return_conditional_losses_597115

inputs*
batch_normalization_2_597061:*
batch_normalization_2_597063:*
batch_normalization_2_597065:*
batch_normalization_2_597067:)
conv2d_6_597070: 
conv2d_6_597072: *
conv2d_7_597076: А
conv2d_7_597078:	А+
conv2d_8_597082:АА
conv2d_8_597084:	А#
dense_6_597090:АвА
dense_6_597092:	А"
dense_7_597096:
АА
dense_7_597098:	А
identityИв-batch_normalization_2/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв conv2d_7/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallвdense_6/StatefulPartitionedCallв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpвdense_7/StatefulPartitionedCallв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpв!dropout_6/StatefulPartitionedCallв!dropout_7/StatefulPartitionedCallв!dropout_8/StatefulPartitionedCall└
lambda_2/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_597019Е
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0batch_normalization_2_597061batch_normalization_2_597063batch_normalization_2_597065batch_normalization_2_597067*
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
GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_596632и
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_6_597070conv2d_6_597072*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_596720ё
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_596652Ы
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_597076conv2d_7_597078*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_596738Є
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_596664Ы
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_597082conv2d_8_597084*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_596756Є
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_596676ї
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_596970р
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_596776Й
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_597090dense_6_597092*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_596793С
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_596931С
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_7_597096dense_7_597098*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_596821С
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_596898Й
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_597070*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_597090*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Б
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_7_597096* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity*dropout_8/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ай
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_7/StatefulPartitionedCall1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╠
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_596601

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
├f
█
__inference_call_530729

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АH
3sequential_2_dense_6_matmul_readvariableop_resource:АвАC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpвBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в1sequential_2/batch_normalization_2/ReadVariableOpв3sequential_2/batch_normalization_2/ReadVariableOp_1в,sequential_2/conv2d_6/BiasAdd/ReadVariableOpв+sequential_2/conv2d_6/Conv2D/ReadVariableOpв,sequential_2/conv2d_7/BiasAdd/ReadVariableOpв+sequential_2/conv2d_7/Conv2D/ReadVariableOpв,sequential_2/conv2d_8/BiasAdd/ReadVariableOpв+sequential_2/conv2d_8/Conv2D/ReadVariableOpв+sequential_2/dense_6/BiasAdd/ReadVariableOpв*sequential_2/dense_6/MatMul/ReadVariableOpв+sequential_2/dense_7/BiasAdd/ReadVariableOpв*sequential_2/dense_7/MatMul/ReadVariableOpВ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( и
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Ю
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Д
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╞
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
й
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0э
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Я
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0└
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЕ
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╟
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
к
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0э
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Я
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0└
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЕ
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:         А╟
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Х
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:         		Аm
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  о
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*)
_output_shapes
:         Авб
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╡
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         АЗ
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0*(
_output_shapes
:         Аа
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╢
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:         АЗ
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0*(
_output_shapes
:         АЕ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ы
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╩
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╩
Щ
(__inference_dense_6_layer_call_fn_598497

inputs
unknown:АвА
	unknown_0:	А
identityИвStatefulPartitionedCall┘
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_596793p
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
├f
█
__inference_call_529376

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АH
3sequential_2_dense_6_matmul_readvariableop_resource:АвАC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpвBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в1sequential_2/batch_normalization_2/ReadVariableOpв3sequential_2/batch_normalization_2/ReadVariableOp_1в,sequential_2/conv2d_6/BiasAdd/ReadVariableOpв+sequential_2/conv2d_6/Conv2D/ReadVariableOpв,sequential_2/conv2d_7/BiasAdd/ReadVariableOpв+sequential_2/conv2d_7/Conv2D/ReadVariableOpв,sequential_2/conv2d_8/BiasAdd/ReadVariableOpв+sequential_2/conv2d_8/Conv2D/ReadVariableOpв+sequential_2/dense_6/BiasAdd/ReadVariableOpв*sequential_2/dense_6/MatMul/ReadVariableOpв+sequential_2/dense_7/BiasAdd/ReadVariableOpв*sequential_2/dense_7/MatMul/ReadVariableOpВ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ─
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskи
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Л
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( и
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ў
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Ю
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK Д
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         KK ╞
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
й
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0э
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Я
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0└
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЕ
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╟
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
к
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0э
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Я
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0└
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЕ
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:         А╟
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Х
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:         		Аm
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  о
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*)
_output_shapes
:         Авб
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0╡
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         АЗ
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0*(
_output_shapes
:         Аа
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╢
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:         АЗ
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0*(
_output_shapes
:         АЕ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ы
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╩
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╦
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_598488

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
Л
 
D__inference_conv2d_7_layer_call_and_return_conditional_losses_596738

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
Х
c
*__inference_dropout_6_layer_call_fn_598460

inputs
identityИвStatefulPartitionedCall╔
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
GPU 2J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_596970x
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
У
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_596652

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
П
А
D__inference_conv2d_8_layer_call_and_return_conditional_losses_598440

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
г
F
*__inference_dropout_7_layer_call_fn_598517

inputs
identity▒
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
GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_596804a
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
Е
▒
D__inference_conv2d_6_layer_call_and_return_conditional_losses_598380

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp|
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
:         KK Ш
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         KK л
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┘
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_598286

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
┘O
ь
H__inference_sequential_2_layer_call_and_return_conditional_losses_597295
lambda_2_input*
batch_normalization_2_597241:*
batch_normalization_2_597243:*
batch_normalization_2_597245:*
batch_normalization_2_597247:)
conv2d_6_597250: 
conv2d_6_597252: *
conv2d_7_597256: А
conv2d_7_597258:	А+
conv2d_8_597262:АА
conv2d_8_597264:	А#
dense_6_597270:АвА
dense_6_597272:	А"
dense_7_597276:
АА
dense_7_597278:	А
identityИв-batch_normalization_2/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв conv2d_7/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallвdense_6/StatefulPartitionedCallв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpвdense_7/StatefulPartitionedCallв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpв!dropout_6/StatefulPartitionedCallв!dropout_7/StatefulPartitionedCallв!dropout_8/StatefulPartitionedCall╚
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_597019Е
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0batch_normalization_2_597241batch_normalization_2_597243batch_normalization_2_597245batch_normalization_2_597247*
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
GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_596632и
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_6_597250conv2d_6_597252*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_596720ё
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_596652Ы
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_597256conv2d_7_597258*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_596738Є
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_596664Ы
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_597262conv2d_8_597264*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_596756Є
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_596676ї
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_596970р
flatten_2/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_596776Й
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_597270dense_6_597272*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_596793С
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_596931С
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_7_597276dense_7_597278*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_596821С
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_596898Й
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_597250*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_597270*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Б
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_7_597276* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity*dropout_8/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ай
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_7/StatefulPartitionedCall1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_2_input
╛
о
$__inference_CNN_layer_call_fn_597396
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
identityИвStatefulPartitionedCallР
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
GPU 2J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_597361o
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
Т

d
E__inference_dropout_7_layer_call_and_return_conditional_losses_596931

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
Ь
л
C__inference_dense_6_layer_call_and_return_conditional_losses_598512

inputs3
matmul_readvariableop_resource:АвА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpw
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
:         АТ
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ак
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         Ав: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:Q M
)
_output_shapes
:         Ав
 
_user_specified_nameinputs
╜
E
)__inference_lambda_2_layer_call_fn_598278

inputs
identity╖
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
GPU 2J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_597019h
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
Ч
к
C__inference_dense_7_layer_call_and_return_conditional_losses_598563

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpv
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
:         АС
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Ак
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ю
а
)__inference_conv2d_7_layer_call_fn_598399

inputs"
unknown: А
	unknown_0:	А
identityИвStatefulPartitionedCallт
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_596738x
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
╩

d
E__inference_dropout_6_layer_call_and_return_conditional_losses_598477

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
╟
Ш
(__inference_dense_7_layer_call_fn_598548

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall┘
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_596821p
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
М
Е
-__inference_sequential_2_layer_call_fn_597179
lambda_2_input
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
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_597115p
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
_user_specified_namelambda_2_input
№
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_598465

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
┘
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_597019

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
╡
F
*__inference_flatten_2_layer_call_fn_598482

inputs
identity▓
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
GPU 2J 8В *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_596776b
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
Ц	
▒
__inference_loss_fn_2_598617M
9dense_7_kernel_regularizer_l2loss_readvariableop_resource:
АА
identityИв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpм
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp9dense_7_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentity"dense_7/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: y
NoOpNoOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp
ї
c
*__inference_dropout_7_layer_call_fn_598522

inputs
identityИвStatefulPartitionedCall┴
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
GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_596931p
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
О%
е
?__inference_CNN_layer_call_and_return_conditional_losses_597617
input_1!
sequential_2_597570:!
sequential_2_597572:!
sequential_2_597574:!
sequential_2_597576:-
sequential_2_597578: !
sequential_2_597580: .
sequential_2_597582: А"
sequential_2_597584:	А/
sequential_2_597586:АА"
sequential_2_597588:	А(
sequential_2_597590:АвА"
sequential_2_597592:	А'
sequential_2_597594:
АА"
sequential_2_597596:	А!
dense_8_597599:	А
dense_8_597601:
identityИв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpвdense_8/StatefulPartitionedCallв$sequential_2/StatefulPartitionedCallЦ
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_2_597570sequential_2_597572sequential_2_597574sequential_2_597576sequential_2_597578sequential_2_597580sequential_2_597582sequential_2_597584sequential_2_597586sequential_2_597588sequential_2_597590sequential_2_597592sequential_2_597594sequential_2_597596*
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_596847У
dense_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0dense_8_597599dense_8_597601*
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
GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_597342Н
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597578*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: З
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597590*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597594* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_8/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
Ж
└
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_596632

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
┘
`
D__inference_lambda_2_layer_call_and_return_conditional_losses_598294

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
╬_
З
H__inference_sequential_2_layer_call_and_return_conditional_losses_598150

inputs;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: B
'conv2d_7_conv2d_readvariableop_resource: А7
(conv2d_7_biasadd_readvariableop_resource:	АC
'conv2d_8_conv2d_readvariableop_resource:АА7
(conv2d_8_biasadd_readvariableop_resource:	А;
&dense_6_matmul_readvariableop_resource:АвА6
'dense_6_biasadd_readvariableop_resource:	А:
&dense_7_matmul_readvariableop_resource:
АА6
'dense_7_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_2/FusedBatchNormV3/ReadVariableOpв7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_2/ReadVariableOpв&batch_normalization_2/ReadVariableOp_1вconv2d_6/BiasAdd/ReadVariableOpвconv2d_6/Conv2D/ReadVariableOpв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpвconv2d_7/BiasAdd/ReadVariableOpвconv2d_7/Conv2D/ReadVariableOpвconv2d_8/BiasAdd/ReadVariableOpвconv2d_8/Conv2D/ReadVariableOpвdense_6/BiasAdd/ReadVariableOpвdense_6/MatMul/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpвdense_7/BiasAdd/ReadVariableOpвdense_7/MatMul/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpu
lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                w
lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            Р
lambda_2/strided_sliceStridedSliceinputs%lambda_2/strided_slice/stack:output:0'lambda_2/strided_slice/stack_1:output:0'lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_maskО
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╜
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3lambda_2/strided_slice:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( О
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0╧
conv2d_6/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         KK м
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
П
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0╞
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Е
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аk
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ан
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
Р
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╞
conv2d_8/Conv2DConv2D max_pooling2d_7/MaxPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Е
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аk
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:         Ан
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
{
dropout_6/IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:         		А`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  З
flatten_2/ReshapeReshapedropout_6/Identity:output:0flatten_2/Const:output:0*
T0*)
_output_shapes
:         АвЗ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0О
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         Аm
dropout_7/IdentityIdentitydense_6/Relu:activations:0*
T0*(
_output_shapes
:         АЖ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0П
dense_7/MatMulMatMuldropout_7/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:         Аm
dropout_8/IdentityIdentitydense_7/Relu:activations:0*
T0*(
_output_shapes
:         Аб
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ъ
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Щ
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: k
IdentityIdentitydropout_8/Identity:output:0^NoOp*
T0*(
_output_shapes
:         Аэ
NoOpNoOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╬d
█
__inference_call_530657

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АH
3sequential_2_dense_6_matmul_readvariableop_resource:АвАC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpвBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в1sequential_2/batch_normalization_2/ReadVariableOpв3sequential_2/batch_normalization_2/ReadVariableOp_1в,sequential_2/conv2d_6/BiasAdd/ReadVariableOpв+sequential_2/conv2d_6/Conv2D/ReadVariableOpв,sequential_2/conv2d_7/BiasAdd/ReadVariableOpв+sequential_2/conv2d_7/Conv2D/ReadVariableOpв,sequential_2/conv2d_8/BiasAdd/ReadVariableOpв+sequential_2/conv2d_8/Conv2D/ReadVariableOpв+sequential_2/dense_6/BiasAdd/ReadVariableOpв*sequential_2/dense_6/MatMul/ReadVariableOpв+sequential_2/dense_7/BiasAdd/ReadVariableOpв*sequential_2/dense_7/MatMul/ReadVariableOpВ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ╝
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_maskи
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Г
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( и
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ю
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
Ю
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╖
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK |
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:АKK ╛
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
й
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0х
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
Я
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╕
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А}
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А┐
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
к
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0х
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
Я
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╕
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА}
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*(
_output_shapes
:АА┐
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
Н
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*(
_output_shapes
:А		Аm
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  ж
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*!
_output_shapes
:ААвб
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0н
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЭ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0о
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААs
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0* 
_output_shapes
:
АА
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0* 
_output_shapes
:
ААа
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0о
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЭ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0о
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААs
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0* 
_output_shapes
:
АА
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0* 
_output_shapes
:
ААЕ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0У
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	АВ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А^
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*
_output_shapes
:	А`
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*
_output_shapes
:	А╩
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
М%
е
?__inference_CNN_layer_call_and_return_conditional_losses_597667
input_1!
sequential_2_597620:!
sequential_2_597622:!
sequential_2_597624:!
sequential_2_597626:-
sequential_2_597628: !
sequential_2_597630: .
sequential_2_597632: А"
sequential_2_597634:	А/
sequential_2_597636:АА"
sequential_2_597638:	А(
sequential_2_597640:АвА"
sequential_2_597642:	А'
sequential_2_597644:
АА"
sequential_2_597646:	А!
dense_8_597649:	А
dense_8_597651:
identityИв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpвdense_8/StatefulPartitionedCallв$sequential_2/StatefulPartitionedCallФ
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_2_597620sequential_2_597622sequential_2_597624sequential_2_597626sequential_2_597628sequential_2_597630sequential_2_597632sequential_2_597634sequential_2_597636sequential_2_597638sequential_2_597640sequential_2_597642sequential_2_597644sequential_2_597646*
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_597115У
dense_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0dense_8_597649dense_8_597651*
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
GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_597342Н
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597628*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: З
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597640*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597644* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_8/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
№
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_596768

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
г

ї
C__inference_dense_8_layer_call_and_return_conditional_losses_598268

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
ї
c
*__inference_dropout_8_layer_call_fn_598573

inputs
identityИвStatefulPartitionedCall┴
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
GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_596898p
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
├
Ц
(__inference_dense_8_layer_call_fn_598257

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall╪
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
GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_597342o
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
╝
о
$__inference_CNN_layer_call_fn_597567
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
identityИвStatefulPartitionedCallО
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
GPU 2J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_597495o
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
╦
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_596776

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
Ї
¤
-__inference_sequential_2_layer_call_fn_598073

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
identityИвStatefulPartitionedCall√
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_597115p
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
░	
╣
__inference_loss_fn_0_598599T
:conv2d_6_kernel_regularizer_l2loss_readvariableop_resource: 
identityИв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp┤
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:conv2d_6_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv2d_6/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp
Л
 
D__inference_conv2d_7_layer_call_and_return_conditional_losses_598410

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
а
о
$__inference_signature_wrapper_597720
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
identityИвStatefulPartitionedCallЄ
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
GPU 2J 8В **
f%R#
!__inference__wrapped_model_596579o
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
╠
Ь
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_598338

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
Л%
д
?__inference_CNN_layer_call_and_return_conditional_losses_597361

inputs!
sequential_2_597302:!
sequential_2_597304:!
sequential_2_597306:!
sequential_2_597308:-
sequential_2_597310: !
sequential_2_597312: .
sequential_2_597314: А"
sequential_2_597316:	А/
sequential_2_597318:АА"
sequential_2_597320:	А(
sequential_2_597322:АвА"
sequential_2_597324:	А'
sequential_2_597326:
АА"
sequential_2_597328:	А!
dense_8_597343:	А
dense_8_597345:
identityИв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpвdense_8/StatefulPartitionedCallв$sequential_2/StatefulPartitionedCallХ
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinputssequential_2_597302sequential_2_597304sequential_2_597306sequential_2_597308sequential_2_597310sequential_2_597312sequential_2_597314sequential_2_597316sequential_2_597318sequential_2_597320sequential_2_597322sequential_2_597324sequential_2_597326sequential_2_597328*
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_596847У
dense_8/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0dense_8_597343dense_8_597345*
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
GPU 2J 8В *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_597342Н
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597310*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: З
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597322*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpsequential_2_597326* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_8/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
г
F
*__inference_dropout_8_layer_call_fn_598568

inputs
identity▒
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
GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_596832a
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
▄
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_598527

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
ЫK
А
H__inference_sequential_2_layer_call_and_return_conditional_losses_597237
lambda_2_input*
batch_normalization_2_597183:*
batch_normalization_2_597185:*
batch_normalization_2_597187:*
batch_normalization_2_597189:)
conv2d_6_597192: 
conv2d_6_597194: *
conv2d_7_597198: А
conv2d_7_597200:	А+
conv2d_8_597204:АА
conv2d_8_597206:	А#
dense_6_597212:АвА
dense_6_597214:	А"
dense_7_597218:
АА
dense_7_597220:	А
identityИв-batch_normalization_2/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallв1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpв conv2d_7/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallвdense_6/StatefulPartitionedCallв0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpвdense_7/StatefulPartitionedCallв0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp╚
lambda_2/PartitionedCallPartitionedCalllambda_2_input*
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
GPU 2J 8В *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_596694З
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0batch_normalization_2_597183batch_normalization_2_597185batch_normalization_2_597187batch_normalization_2_597189*
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
GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_596601и
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_6_597192conv2d_6_597194*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_596720ё
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_596652Ы
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_597198conv2d_7_597200*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_596738Є
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_596664Ы
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_597204conv2d_8_597206*
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
GPU 2J 8В *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_596756Є
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_596676х
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_596768╪
flatten_2/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_596776Й
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_597212dense_6_597214*
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
GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_596793▌
dropout_7/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_596804Й
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_7_597218dense_7_597220*
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
GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_596821▌
dropout_8/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8В *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_596832Й
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_6_597192*&
_output_shapes
: *
dtype0И
"conv2d_6/kernel/Regularizer/L2LossL2Loss9conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<а
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0+conv2d_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_6_597212*!
_output_shapes
:АвА*
dtype0Ж
!dense_6/kernel/Regularizer/L2LossL2Loss8dense_6/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0*dense_6/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Б
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_7_597218* 
_output_shapes
:
АА*
dtype0Ж
!dense_7/kernel/Regularizer/L2LossL2Loss8dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: e
 dense_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<Э
dense_7/kernel/Regularizer/mulMul)dense_7/kernel/Regularizer/mul/x:output:0*dense_7/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: r
IdentityIdentity"dropout_8/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А╜
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall2^conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/L2Loss/ReadVariableOp ^dense_7/StatefulPartitionedCall1^dense_7/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2f
1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp1conv2d_6/kernel/Regularizer/L2Loss/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2d
0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp0dense_6/kernel/Regularizer/L2Loss/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2d
0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp0dense_7/kernel/Regularizer/L2Loss/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_2_input
╗
н
$__inference_CNN_layer_call_fn_597757

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
identityИвStatefulPartitionedCallП
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
GPU 2J 8В *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_597361o
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
╬d
█
__inference_call_530585

inputsH
:sequential_2_batch_normalization_2_readvariableop_resource:J
<sequential_2_batch_normalization_2_readvariableop_1_resource:Y
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:[
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_2_conv2d_6_conv2d_readvariableop_resource: C
5sequential_2_conv2d_6_biasadd_readvariableop_resource: O
4sequential_2_conv2d_7_conv2d_readvariableop_resource: АD
5sequential_2_conv2d_7_biasadd_readvariableop_resource:	АP
4sequential_2_conv2d_8_conv2d_readvariableop_resource:ААD
5sequential_2_conv2d_8_biasadd_readvariableop_resource:	АH
3sequential_2_dense_6_matmul_readvariableop_resource:АвАC
4sequential_2_dense_6_biasadd_readvariableop_resource:	АG
3sequential_2_dense_7_matmul_readvariableop_resource:
ААC
4sequential_2_dense_7_biasadd_readvariableop_resource:	А9
&dense_8_matmul_readvariableop_resource:	А5
'dense_8_biasadd_readvariableop_resource:
identityИвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpвBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpвDsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1в1sequential_2/batch_normalization_2/ReadVariableOpв3sequential_2/batch_normalization_2/ReadVariableOp_1в,sequential_2/conv2d_6/BiasAdd/ReadVariableOpв+sequential_2/conv2d_6/Conv2D/ReadVariableOpв,sequential_2/conv2d_7/BiasAdd/ReadVariableOpв+sequential_2/conv2d_7/Conv2D/ReadVariableOpв,sequential_2/conv2d_8/BiasAdd/ReadVariableOpв+sequential_2/conv2d_8/Conv2D/ReadVariableOpв+sequential_2/dense_6/BiasAdd/ReadVariableOpв*sequential_2/dense_6/MatMul/ReadVariableOpв+sequential_2/dense_7/BiasAdd/ReadVariableOpв*sequential_2/dense_7/MatMul/ReadVariableOpВ
)sequential_2/lambda_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                Д
+sequential_2/lambda_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            ╝
#sequential_2/lambda_2/strided_sliceStridedSliceinputs2sequential_2/lambda_2/strided_slice/stack:output:04sequential_2/lambda_2/strided_slice/stack_1:output:04sequential_2/lambda_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_maskи
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0м
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0╩
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0╬
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Г
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,sequential_2/lambda_2/strided_slice:output:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( и
+sequential_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ю
sequential_2/conv2d_6/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
Ю
,sequential_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╖
sequential_2/conv2d_6/BiasAddBiasAdd%sequential_2/conv2d_6/Conv2D:output:04sequential_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK |
sequential_2/conv2d_6/ReluRelu&sequential_2/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:АKK ╛
$sequential_2/max_pooling2d_6/MaxPoolMaxPool(sequential_2/conv2d_6/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
й
+sequential_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype0х
sequential_2/conv2d_7/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:03sequential_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
Я
,sequential_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╕
sequential_2/conv2d_7/BiasAddBiasAdd%sequential_2/conv2d_7/Conv2D:output:04sequential_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А}
sequential_2/conv2d_7/ReluRelu&sequential_2/conv2d_7/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А┐
$sequential_2/max_pooling2d_7/MaxPoolMaxPool(sequential_2/conv2d_7/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
к
+sequential_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0х
sequential_2/conv2d_8/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:03sequential_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
Я
,sequential_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╕
sequential_2/conv2d_8/BiasAddBiasAdd%sequential_2/conv2d_8/Conv2D:output:04sequential_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА}
sequential_2/conv2d_8/ReluRelu&sequential_2/conv2d_8/BiasAdd:output:0*
T0*(
_output_shapes
:АА┐
$sequential_2/max_pooling2d_8/MaxPoolMaxPool(sequential_2/conv2d_8/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
Н
sequential_2/dropout_6/IdentityIdentity-sequential_2/max_pooling2d_8/MaxPool:output:0*
T0*(
_output_shapes
:А		Аm
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  ж
sequential_2/flatten_2/ReshapeReshape(sequential_2/dropout_6/Identity:output:0%sequential_2/flatten_2/Const:output:0*
T0*!
_output_shapes
:ААвб
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource*!
_output_shapes
:АвА*
dtype0н
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЭ
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0о
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААs
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0* 
_output_shapes
:
АА
sequential_2/dropout_7/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0* 
_output_shapes
:
ААа
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0о
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_7/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААЭ
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0о
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААs
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0* 
_output_shapes
:
АА
sequential_2/dropout_8/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0* 
_output_shapes
:
ААЕ
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0У
dense_8/MatMulMatMul(sequential_2/dropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	АВ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ж
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А^
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*
_output_shapes
:	А`
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*
_output_shapes
:	А╩
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOpC^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpE^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12^sequential_2/batch_normalization_2/ReadVariableOp4^sequential_2/batch_normalization_2/ReadVariableOp_1-^sequential_2/conv2d_6/BiasAdd/ReadVariableOp,^sequential_2/conv2d_6/Conv2D/ReadVariableOp-^sequential_2/conv2d_7/BiasAdd/ReadVariableOp,^sequential_2/conv2d_7/Conv2D/ReadVariableOp-^sequential_2/conv2d_8/BiasAdd/ReadVariableOp,^sequential_2/conv2d_8/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2И
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpBsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2М
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12f
1sequential_2/batch_normalization_2/ReadVariableOp1sequential_2/batch_normalization_2/ReadVariableOp2j
3sequential_2/batch_normalization_2/ReadVariableOp_13sequential_2/batch_normalization_2/ReadVariableOp_12\
,sequential_2/conv2d_6/BiasAdd/ReadVariableOp,sequential_2/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_6/Conv2D/ReadVariableOp+sequential_2/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_7/BiasAdd/ReadVariableOp,sequential_2/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_7/Conv2D/ReadVariableOp+sequential_2/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_8/BiasAdd/ReadVariableOp,sequential_2/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_8/Conv2D/ReadVariableOp+sequential_2/conv2d_8/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
г

ї
C__inference_dense_8_layer_call_and_return_conditional_losses_597342

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
О
Е
-__inference_sequential_2_layer_call_fn_596878
lambda_2_input
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
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCalllambda_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_596847p
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
_user_specified_namelambda_2_input
М_
Ы
__inference__traced_save_598790
file_prefix:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_batch_normalization_2_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_2_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_2_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_2_beta_read_readvariableop5
1savev2_adam_m_conv2d_6_kernel_read_readvariableop5
1savev2_adam_v_conv2d_6_kernel_read_readvariableop3
/savev2_adam_m_conv2d_6_bias_read_readvariableop3
/savev2_adam_v_conv2d_6_bias_read_readvariableop5
1savev2_adam_m_conv2d_7_kernel_read_readvariableop5
1savev2_adam_v_conv2d_7_kernel_read_readvariableop3
/savev2_adam_m_conv2d_7_bias_read_readvariableop3
/savev2_adam_v_conv2d_7_bias_read_readvariableop5
1savev2_adam_m_conv2d_8_kernel_read_readvariableop5
1savev2_adam_v_conv2d_8_kernel_read_readvariableop3
/savev2_adam_m_conv2d_8_bias_read_readvariableop3
/savev2_adam_v_conv2d_8_bias_read_readvariableop4
0savev2_adam_m_dense_6_kernel_read_readvariableop4
0savev2_adam_v_dense_6_kernel_read_readvariableop2
.savev2_adam_m_dense_6_bias_read_readvariableop2
.savev2_adam_v_dense_6_bias_read_readvariableop4
0savev2_adam_m_dense_7_kernel_read_readvariableop4
0savev2_adam_v_dense_7_kernel_read_readvariableop2
.savev2_adam_m_dense_7_bias_read_readvariableop2
.savev2_adam_v_dense_7_bias_read_readvariableop4
0savev2_adam_m_dense_8_kernel_read_readvariableop4
0savev2_adam_v_dense_8_kernel_read_readvariableop2
.savev2_adam_m_dense_8_bias_read_readvariableop2
.savev2_adam_v_dense_8_bias_read_readvariableop&
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
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ё
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_batch_normalization_2_gamma_read_readvariableop=savev2_adam_v_batch_normalization_2_gamma_read_readvariableop<savev2_adam_m_batch_normalization_2_beta_read_readvariableop<savev2_adam_v_batch_normalization_2_beta_read_readvariableop1savev2_adam_m_conv2d_6_kernel_read_readvariableop1savev2_adam_v_conv2d_6_kernel_read_readvariableop/savev2_adam_m_conv2d_6_bias_read_readvariableop/savev2_adam_v_conv2d_6_bias_read_readvariableop1savev2_adam_m_conv2d_7_kernel_read_readvariableop1savev2_adam_v_conv2d_7_kernel_read_readvariableop/savev2_adam_m_conv2d_7_bias_read_readvariableop/savev2_adam_v_conv2d_7_bias_read_readvariableop1savev2_adam_m_conv2d_8_kernel_read_readvariableop1savev2_adam_v_conv2d_8_kernel_read_readvariableop/savev2_adam_m_conv2d_8_bias_read_readvariableop/savev2_adam_v_conv2d_8_bias_read_readvariableop0savev2_adam_m_dense_6_kernel_read_readvariableop0savev2_adam_v_dense_6_kernel_read_readvariableop.savev2_adam_m_dense_6_bias_read_readvariableop.savev2_adam_v_dense_6_bias_read_readvariableop0savev2_adam_m_dense_7_kernel_read_readvariableop0savev2_adam_v_dense_7_kernel_read_readvariableop.savev2_adam_m_dense_7_bias_read_readvariableop.savev2_adam_v_dense_7_bias_read_readvariableop0savev2_adam_m_dense_8_kernel_read_readvariableop0savev2_adam_v_dense_8_kernel_read_readvariableop.savev2_adam_m_dense_8_bias_read_readvariableop.savev2_adam_v_dense_8_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
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
: "Ж
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
StatefulPartitionedCall:0         tensorflow/serving/predict:Б└
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
╣
"trace_0
#trace_1
$trace_2
%trace_32╬
$__inference_CNN_layer_call_fn_597396
$__inference_CNN_layer_call_fn_597757
$__inference_CNN_layer_call_fn_597794
$__inference_CNN_layer_call_fn_597567│
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
е
&trace_0
'trace_1
(trace_2
)trace_32║
?__inference_CNN_layer_call_and_return_conditional_losses_597878
?__inference_CNN_layer_call_and_return_conditional_losses_597983
?__inference_CNN_layer_call_and_return_conditional_losses_597617
?__inference_CNN_layer_call_and_return_conditional_losses_597667│
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
╠B╔
!__inference__wrapped_model_596579input_1"Ш
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
╥
Ktrace_0
Ltrace_1
Mtrace_22Б
__inference_call_530585
__inference_call_530657
__inference_call_530729│
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
):'2batch_normalization_2/gamma
(:&2batch_normalization_2/beta
1:/ (2!batch_normalization_2/moving_mean
5:3 (2%batch_normalization_2/moving_variance
):' 2conv2d_6/kernel
: 2conv2d_6/bias
*:( А2conv2d_7/kernel
:А2conv2d_7/bias
+:)АА2conv2d_8/kernel
:А2conv2d_8/bias
#:!АвА2dense_6/kernel
:А2dense_6/bias
": 
АА2dense_7/kernel
:А2dense_7/bias
!:	А2dense_8/kernel
:2dense_8/bias
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
ъBч
$__inference_CNN_layer_call_fn_597396input_1"│
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
щBц
$__inference_CNN_layer_call_fn_597757inputs"│
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
щBц
$__inference_CNN_layer_call_fn_597794inputs"│
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
$__inference_CNN_layer_call_fn_597567input_1"│
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
ДBБ
?__inference_CNN_layer_call_and_return_conditional_losses_597878inputs"│
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
ДBБ
?__inference_CNN_layer_call_and_return_conditional_losses_597983inputs"│
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
?__inference_CNN_layer_call_and_return_conditional_losses_597617input_1"│
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
?__inference_CNN_layer_call_and_return_conditional_losses_597667input_1"│
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
ё
┤trace_0
╡trace_1
╢trace_2
╖trace_32■
-__inference_sequential_2_layer_call_fn_596878
-__inference_sequential_2_layer_call_fn_598040
-__inference_sequential_2_layer_call_fn_598073
-__inference_sequential_2_layer_call_fn_597179┐
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
▌
╕trace_0
╣trace_1
║trace_2
╗trace_32ъ
H__inference_sequential_2_layer_call_and_return_conditional_losses_598150
H__inference_sequential_2_layer_call_and_return_conditional_losses_598248
H__inference_sequential_2_layer_call_and_return_conditional_losses_597237
H__inference_sequential_2_layer_call_and_return_conditional_losses_597295┐
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
ю
┴trace_02╧
(__inference_dense_8_layer_call_fn_598257в
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
Й
┬trace_02ъ
C__inference_dense_8_layer_call_and_return_conditional_losses_598268в
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
▄B┘
__inference_call_530585inputs"│
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
▄B┘
__inference_call_530657inputs"│
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
▄B┘
__inference_call_530729inputs"│
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
╦B╚
$__inference_signature_wrapper_597720input_1"Ф
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
╙
эtrace_0
юtrace_12Ш
)__inference_lambda_2_layer_call_fn_598273
)__inference_lambda_2_layer_call_fn_598278┐
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
Й
яtrace_0
Ёtrace_12╬
D__inference_lambda_2_layer_call_and_return_conditional_losses_598286
D__inference_lambda_2_layer_call_and_return_conditional_losses_598294┐
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
с
Ўtrace_0
ўtrace_12ж
6__inference_batch_normalization_2_layer_call_fn_598307
6__inference_batch_normalization_2_layer_call_fn_598320│
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
Ч
°trace_0
∙trace_12▄
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_598338
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_598356│
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
я
 trace_02╨
)__inference_conv2d_6_layer_call_fn_598365в
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
К
Аtrace_02ы
D__inference_conv2d_6_layer_call_and_return_conditional_losses_598380в
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
Ў
Жtrace_02╫
0__inference_max_pooling2d_6_layer_call_fn_598385в
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
С
Зtrace_02Є
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_598390в
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
я
Нtrace_02╨
)__inference_conv2d_7_layer_call_fn_598399в
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
К
Оtrace_02ы
D__inference_conv2d_7_layer_call_and_return_conditional_losses_598410в
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
Ў
Фtrace_02╫
0__inference_max_pooling2d_7_layer_call_fn_598415в
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
С
Хtrace_02Є
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_598420в
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
я
Ыtrace_02╨
)__inference_conv2d_8_layer_call_fn_598429в
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
К
Ьtrace_02ы
D__inference_conv2d_8_layer_call_and_return_conditional_losses_598440в
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
Ў
вtrace_02╫
0__inference_max_pooling2d_8_layer_call_fn_598445в
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
С
гtrace_02Є
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_598450в
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
╔
йtrace_0
кtrace_12О
*__inference_dropout_6_layer_call_fn_598455
*__inference_dropout_6_layer_call_fn_598460│
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
 
лtrace_0
мtrace_12─
E__inference_dropout_6_layer_call_and_return_conditional_losses_598465
E__inference_dropout_6_layer_call_and_return_conditional_losses_598477│
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
Ё
▓trace_02╤
*__inference_flatten_2_layer_call_fn_598482в
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
Л
│trace_02ь
E__inference_flatten_2_layer_call_and_return_conditional_losses_598488в
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
ю
╣trace_02╧
(__inference_dense_6_layer_call_fn_598497в
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
Й
║trace_02ъ
C__inference_dense_6_layer_call_and_return_conditional_losses_598512в
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
╔
└trace_0
┴trace_12О
*__inference_dropout_7_layer_call_fn_598517
*__inference_dropout_7_layer_call_fn_598522│
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
 
┬trace_0
├trace_12─
E__inference_dropout_7_layer_call_and_return_conditional_losses_598527
E__inference_dropout_7_layer_call_and_return_conditional_losses_598539│
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
ю
╔trace_02╧
(__inference_dense_7_layer_call_fn_598548в
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
Й
╩trace_02ъ
C__inference_dense_7_layer_call_and_return_conditional_losses_598563в
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
╔
╨trace_0
╤trace_12О
*__inference_dropout_8_layer_call_fn_598568
*__inference_dropout_8_layer_call_fn_598573│
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
 
╥trace_0
╙trace_12─
E__inference_dropout_8_layer_call_and_return_conditional_losses_598578
E__inference_dropout_8_layer_call_and_return_conditional_losses_598590│
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
╧
╘trace_02░
__inference_loss_fn_0_598599П
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
╧
╒trace_02░
__inference_loss_fn_1_598608П
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
╧
╓trace_02░
__inference_loss_fn_2_598617П
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
ЖBГ
-__inference_sequential_2_layer_call_fn_596878lambda_2_input"┐
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
■B√
-__inference_sequential_2_layer_call_fn_598040inputs"┐
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
■B√
-__inference_sequential_2_layer_call_fn_598073inputs"┐
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
ЖBГ
-__inference_sequential_2_layer_call_fn_597179lambda_2_input"┐
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
ЩBЦ
H__inference_sequential_2_layer_call_and_return_conditional_losses_598150inputs"┐
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
ЩBЦ
H__inference_sequential_2_layer_call_and_return_conditional_losses_598248inputs"┐
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
бBЮ
H__inference_sequential_2_layer_call_and_return_conditional_losses_597237lambda_2_input"┐
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
бBЮ
H__inference_sequential_2_layer_call_and_return_conditional_losses_597295lambda_2_input"┐
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
▄B┘
(__inference_dense_8_layer_call_fn_598257inputs"в
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
ўBЇ
C__inference_dense_8_layer_call_and_return_conditional_losses_598268inputs"в
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
.:,2"Adam/m/batch_normalization_2/gamma
.:,2"Adam/v/batch_normalization_2/gamma
-:+2!Adam/m/batch_normalization_2/beta
-:+2!Adam/v/batch_normalization_2/beta
.:, 2Adam/m/conv2d_6/kernel
.:, 2Adam/v/conv2d_6/kernel
 : 2Adam/m/conv2d_6/bias
 : 2Adam/v/conv2d_6/bias
/:- А2Adam/m/conv2d_7/kernel
/:- А2Adam/v/conv2d_7/kernel
!:А2Adam/m/conv2d_7/bias
!:А2Adam/v/conv2d_7/bias
0:.АА2Adam/m/conv2d_8/kernel
0:.АА2Adam/v/conv2d_8/kernel
!:А2Adam/m/conv2d_8/bias
!:А2Adam/v/conv2d_8/bias
(:&АвА2Adam/m/dense_6/kernel
(:&АвА2Adam/v/dense_6/kernel
 :А2Adam/m/dense_6/bias
 :А2Adam/v/dense_6/bias
':%
АА2Adam/m/dense_7/kernel
':%
АА2Adam/v/dense_7/kernel
 :А2Adam/m/dense_7/bias
 :А2Adam/v/dense_7/bias
&:$	А2Adam/m/dense_8/kernel
&:$	А2Adam/v/dense_8/kernel
:2Adam/m/dense_8/bias
:2Adam/v/dense_8/bias
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
·Bў
)__inference_lambda_2_layer_call_fn_598273inputs"┐
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
·Bў
)__inference_lambda_2_layer_call_fn_598278inputs"┐
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
ХBТ
D__inference_lambda_2_layer_call_and_return_conditional_losses_598286inputs"┐
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
ХBТ
D__inference_lambda_2_layer_call_and_return_conditional_losses_598294inputs"┐
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
√B°
6__inference_batch_normalization_2_layer_call_fn_598307inputs"│
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
√B°
6__inference_batch_normalization_2_layer_call_fn_598320inputs"│
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
ЦBУ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_598338inputs"│
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
ЦBУ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_598356inputs"│
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
▌B┌
)__inference_conv2d_6_layer_call_fn_598365inputs"в
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
°Bї
D__inference_conv2d_6_layer_call_and_return_conditional_losses_598380inputs"в
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
фBс
0__inference_max_pooling2d_6_layer_call_fn_598385inputs"в
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
 B№
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_598390inputs"в
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
▌B┌
)__inference_conv2d_7_layer_call_fn_598399inputs"в
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
°Bї
D__inference_conv2d_7_layer_call_and_return_conditional_losses_598410inputs"в
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
фBс
0__inference_max_pooling2d_7_layer_call_fn_598415inputs"в
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
 B№
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_598420inputs"в
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
▌B┌
)__inference_conv2d_8_layer_call_fn_598429inputs"в
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
°Bї
D__inference_conv2d_8_layer_call_and_return_conditional_losses_598440inputs"в
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
фBс
0__inference_max_pooling2d_8_layer_call_fn_598445inputs"в
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
 B№
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_598450inputs"в
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
яBь
*__inference_dropout_6_layer_call_fn_598455inputs"│
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
яBь
*__inference_dropout_6_layer_call_fn_598460inputs"│
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
КBЗ
E__inference_dropout_6_layer_call_and_return_conditional_losses_598465inputs"│
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
КBЗ
E__inference_dropout_6_layer_call_and_return_conditional_losses_598477inputs"│
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
▐B█
*__inference_flatten_2_layer_call_fn_598482inputs"в
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_598488inputs"в
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
▄B┘
(__inference_dense_6_layer_call_fn_598497inputs"в
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
ўBЇ
C__inference_dense_6_layer_call_and_return_conditional_losses_598512inputs"в
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
яBь
*__inference_dropout_7_layer_call_fn_598517inputs"│
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
яBь
*__inference_dropout_7_layer_call_fn_598522inputs"│
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
КBЗ
E__inference_dropout_7_layer_call_and_return_conditional_losses_598527inputs"│
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
КBЗ
E__inference_dropout_7_layer_call_and_return_conditional_losses_598539inputs"│
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
▄B┘
(__inference_dense_7_layer_call_fn_598548inputs"в
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
ўBЇ
C__inference_dense_7_layer_call_and_return_conditional_losses_598563inputs"в
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
яBь
*__inference_dropout_8_layer_call_fn_598568inputs"│
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
яBь
*__inference_dropout_8_layer_call_fn_598573inputs"│
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
КBЗ
E__inference_dropout_8_layer_call_and_return_conditional_losses_598578inputs"│
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
КBЗ
E__inference_dropout_8_layer_call_and_return_conditional_losses_598590inputs"│
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
│B░
__inference_loss_fn_0_598599"П
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
│B░
__inference_loss_fn_1_598608"П
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
│B░
__inference_loss_fn_2_598617"П
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
annotationsк *в ┴
?__inference_CNN_layer_call_and_return_conditional_losses_597617~<в9
2в/
)К&
input_1         KK
p 
к ",в)
"К
tensor_0         
Ъ ┴
?__inference_CNN_layer_call_and_return_conditional_losses_597667~<в9
2в/
)К&
input_1         KK
p
к ",в)
"К
tensor_0         
Ъ └
?__inference_CNN_layer_call_and_return_conditional_losses_597878};в8
1в.
(К%
inputs         KK
p 
к ",в)
"К
tensor_0         
Ъ └
?__inference_CNN_layer_call_and_return_conditional_losses_597983};в8
1в.
(К%
inputs         KK
p
к ",в)
"К
tensor_0         
Ъ Ы
$__inference_CNN_layer_call_fn_597396s<в9
2в/
)К&
input_1         KK
p 
к "!К
unknown         Ы
$__inference_CNN_layer_call_fn_597567s<в9
2в/
)К&
input_1         KK
p
к "!К
unknown         Ъ
$__inference_CNN_layer_call_fn_597757r;в8
1в.
(К%
inputs         KK
p 
к "!К
unknown         Ъ
$__inference_CNN_layer_call_fn_597794r;в8
1в.
(К%
inputs         KK
p
к "!К
unknown         з
!__inference__wrapped_model_596579Б8в5
.в+
)К&
input_1         KK
к "3к0
.
output_1"К
output_1         є
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_598338ЭMвJ
Cв@
:К7
inputs+                           
p 
к "FвC
<К9
tensor_0+                           
Ъ є
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_598356ЭMвJ
Cв@
:К7
inputs+                           
p
к "FвC
<К9
tensor_0+                           
Ъ ═
6__inference_batch_normalization_2_layer_call_fn_598307ТMвJ
Cв@
:К7
inputs+                           
p 
к ";К8
unknown+                           ═
6__inference_batch_normalization_2_layer_call_fn_598320ТMвJ
Cв@
:К7
inputs+                           
p
к ";К8
unknown+                           }
__inference_call_530585b3в0
)в&
 К
inputsАKK
p
к "К
unknown	А}
__inference_call_530657b3в0
)в&
 К
inputsАKK
p 
к "К
unknown	АН
__inference_call_530729r;в8
1в.
(К%
inputs         KK
p 
к "!К
unknown         ╗
D__inference_conv2d_6_layer_call_and_return_conditional_losses_598380s7в4
-в*
(К%
inputs         KK
к "4в1
*К'
tensor_0         KK 
Ъ Х
)__inference_conv2d_6_layer_call_fn_598365h7в4
-в*
(К%
inputs         KK
к ")К&
unknown         KK ╝
D__inference_conv2d_7_layer_call_and_return_conditional_losses_598410t7в4
-в*
(К%
inputs         %% 
к "5в2
+К(
tensor_0         %%А
Ъ Ц
)__inference_conv2d_7_layer_call_fn_598399i7в4
-в*
(К%
inputs         %% 
к "*К'
unknown         %%А╜
D__inference_conv2d_8_layer_call_and_return_conditional_losses_598440u8в5
.в+
)К&
inputs         А
к "5в2
+К(
tensor_0         А
Ъ Ч
)__inference_conv2d_8_layer_call_fn_598429j8в5
.в+
)К&
inputs         А
к "*К'
unknown         Ан
C__inference_dense_6_layer_call_and_return_conditional_losses_598512f1в.
'в$
"К
inputs         Ав
к "-в*
#К 
tensor_0         А
Ъ З
(__inference_dense_6_layer_call_fn_598497[1в.
'в$
"К
inputs         Ав
к ""К
unknown         Ам
C__inference_dense_7_layer_call_and_return_conditional_losses_598563e0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ Ж
(__inference_dense_7_layer_call_fn_598548Z0в-
&в#
!К
inputs         А
к ""К
unknown         Ал
C__inference_dense_8_layer_call_and_return_conditional_losses_598268d0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         
Ъ Е
(__inference_dense_8_layer_call_fn_598257Y0в-
&в#
!К
inputs         А
к "!К
unknown         ╛
E__inference_dropout_6_layer_call_and_return_conditional_losses_598465u<в9
2в/
)К&
inputs         		А
p 
к "5в2
+К(
tensor_0         		А
Ъ ╛
E__inference_dropout_6_layer_call_and_return_conditional_losses_598477u<в9
2в/
)К&
inputs         		А
p
к "5в2
+К(
tensor_0         		А
Ъ Ш
*__inference_dropout_6_layer_call_fn_598455j<в9
2в/
)К&
inputs         		А
p 
к "*К'
unknown         		АШ
*__inference_dropout_6_layer_call_fn_598460j<в9
2в/
)К&
inputs         		А
p
к "*К'
unknown         		Ао
E__inference_dropout_7_layer_call_and_return_conditional_losses_598527e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ о
E__inference_dropout_7_layer_call_and_return_conditional_losses_598539e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ И
*__inference_dropout_7_layer_call_fn_598517Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         АИ
*__inference_dropout_7_layer_call_fn_598522Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         Ао
E__inference_dropout_8_layer_call_and_return_conditional_losses_598578e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ о
E__inference_dropout_8_layer_call_and_return_conditional_losses_598590e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ И
*__inference_dropout_8_layer_call_fn_598568Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         АИ
*__inference_dropout_8_layer_call_fn_598573Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         А│
E__inference_flatten_2_layer_call_and_return_conditional_losses_598488j8в5
.в+
)К&
inputs         		А
к ".в+
$К!
tensor_0         Ав
Ъ Н
*__inference_flatten_2_layer_call_fn_598482_8в5
.в+
)К&
inputs         		А
к "#К 
unknown         Ав┐
D__inference_lambda_2_layer_call_and_return_conditional_losses_598286w?в<
5в2
(К%
inputs         KK

 
p 
к "4в1
*К'
tensor_0         KK
Ъ ┐
D__inference_lambda_2_layer_call_and_return_conditional_losses_598294w?в<
5в2
(К%
inputs         KK

 
p
к "4в1
*К'
tensor_0         KK
Ъ Щ
)__inference_lambda_2_layer_call_fn_598273l?в<
5в2
(К%
inputs         KK

 
p 
к ")К&
unknown         KKЩ
)__inference_lambda_2_layer_call_fn_598278l?в<
5в2
(К%
inputs         KK

 
p
к ")К&
unknown         KKD
__inference_loss_fn_0_598599$в

в 
к "К
unknown D
__inference_loss_fn_1_598608$в

в 
к "К
unknown D
__inference_loss_fn_2_598617$в

в 
к "К
unknown ї
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_598390еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╧
0__inference_max_pooling2d_6_layer_call_fn_598385ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ї
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_598420еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╧
0__inference_max_pooling2d_7_layer_call_fn_598415ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ї
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_598450еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╧
0__inference_max_pooling2d_8_layer_call_fn_598445ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ╒
H__inference_sequential_2_layer_call_and_return_conditional_losses_597237ИGвD
=в:
0К-
lambda_2_input         KK
p 

 
к "-в*
#К 
tensor_0         А
Ъ ╒
H__inference_sequential_2_layer_call_and_return_conditional_losses_597295ИGвD
=в:
0К-
lambda_2_input         KK
p

 
к "-в*
#К 
tensor_0         А
Ъ ═
H__inference_sequential_2_layer_call_and_return_conditional_losses_598150А?в<
5в2
(К%
inputs         KK
p 

 
к "-в*
#К 
tensor_0         А
Ъ ═
H__inference_sequential_2_layer_call_and_return_conditional_losses_598248А?в<
5в2
(К%
inputs         KK
p

 
к "-в*
#К 
tensor_0         А
Ъ о
-__inference_sequential_2_layer_call_fn_596878}GвD
=в:
0К-
lambda_2_input         KK
p 

 
к ""К
unknown         Ао
-__inference_sequential_2_layer_call_fn_597179}GвD
=в:
0К-
lambda_2_input         KK
p

 
к ""К
unknown         Аж
-__inference_sequential_2_layer_call_fn_598040u?в<
5в2
(К%
inputs         KK
p 

 
к ""К
unknown         Аж
-__inference_sequential_2_layer_call_fn_598073u?в<
5в2
(К%
inputs         KK
p

 
к ""К
unknown         А╡
$__inference_signature_wrapper_597720МCв@
в 
9к6
4
input_1)К&
input_1         KK"3к0
.
output_1"К
output_1         