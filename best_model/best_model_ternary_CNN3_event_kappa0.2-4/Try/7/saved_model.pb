Щ·
ф┤
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
·
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
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
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ю╒
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes
:	А*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0
Д
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
: *
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
: *
dtype0
Е
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*!
shared_nameconv2d_29/kernel
~
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*'
_output_shapes
: А*
dtype0
u
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_29/bias
n
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_30/kernel

$conv2d_30/kernel/Read/ReadVariableOpReadVariableOpconv2d_30/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_30/bias
n
"conv2d_30/bias/Read/ReadVariableOpReadVariableOpconv2d_30/bias*
_output_shapes	
:А*
dtype0
Ж
conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*!
shared_nameconv2d_31/kernel

$conv2d_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_31/kernel*(
_output_shapes
:АА*
dtype0
u
conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_31/bias
n
"conv2d_31/bias/Read/ReadVariableOpReadVariableOpconv2d_31/bias*
_output_shapes	
:А*
dtype0
|
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А* 
shared_namedense_14/kernel
u
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel* 
_output_shapes
:
А@А*
dtype0
s
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_14/bias
l
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes	
:А*
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:*
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
Й
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_15/kernel/m
В
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes
:	А*
dtype0
А
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/m
Х
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/m
У
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_28/kernel/m
Л
+Adam/conv2d_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_28/bias/m
{
)Adam/conv2d_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/m*
_output_shapes
: *
dtype0
У
Adam/conv2d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_29/kernel/m
М
+Adam/conv2d_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/m*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_29/bias/m
|
)Adam/conv2d_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_30/kernel/m
Н
+Adam/conv2d_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_30/bias/m
|
)Adam/conv2d_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/bias/m*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_31/kernel/m
Н
+Adam/conv2d_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/m*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_31/bias/m
|
)Adam/conv2d_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_14/kernel/m
Г
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_14/bias/m
z
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_15/kernel/v
В
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes
:	А*
dtype0
А
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/v
Х
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/v
У
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_28/kernel/v
Л
+Adam/conv2d_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_28/bias/v
{
)Adam/conv2d_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/v*
_output_shapes
: *
dtype0
У
Adam/conv2d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: А*(
shared_nameAdam/conv2d_29/kernel/v
М
+Adam/conv2d_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/v*'
_output_shapes
: А*
dtype0
Г
Adam/conv2d_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_29/bias/v
|
)Adam/conv2d_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_30/kernel/v
Н
+Adam/conv2d_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_30/bias/v
|
)Adam/conv2d_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/bias/v*
_output_shapes	
:А*
dtype0
Ф
Adam/conv2d_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*(
shared_nameAdam/conv2d_31/kernel/v
Н
+Adam/conv2d_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/v*(
_output_shapes
:АА*
dtype0
Г
Adam/conv2d_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*&
shared_nameAdam/conv2d_31/bias/v
|
)Adam/conv2d_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_14/kernel/v
Г
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_14/bias/v
z
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
╪`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*У`
valueЙ`BЖ` B _
К

h2ptjl
_output
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
и
	layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer_with_weights-4
layer-8
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
╪
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m╧'m╨(m╤)m╥*m╙+m╘,m╒-m╓.m╫/m╪0m┘1m┌v█v▄&v▌'v▐(v▀)vр*vс+vт,vу-vф.vх/vц0vч1vш
f
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
12
13
v
&0
'1
22
33
(4
)5
*6
+7
,8
-9
.10
/11
012
113
14
15
 
н
trainable_variables
4non_trainable_variables
	variables
regularization_losses
5layer_regularization_losses
6layer_metrics

7layers
8metrics
 
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
Ч
=axis
	&gamma
'beta
2moving_mean
3moving_variance
>trainable_variables
?	variables
@regularization_losses
A	keras_api
h

(kernel
)bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
R
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
h

*kernel
+bias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
R
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
h

,kernel
-bias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
R
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
h

.kernel
/bias
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
R
^trainable_variables
_	variables
`regularization_losses
a	keras_api
R
btrainable_variables
c	variables
dregularization_losses
e	keras_api
R
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
h

0kernel
1bias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
R
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
V
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
f
&0
'1
22
33
(4
)5
*6
+7
,8
-9
.10
/11
012
113
 
н
trainable_variables
rnon_trainable_variables
	variables
regularization_losses
slayer_regularization_losses
tlayer_metrics

ulayers
vmetrics
NL
VARIABLE_VALUEdense_15/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_15/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
н
trainable_variables
wnon_trainable_variables
	variables
regularization_losses
xlayer_regularization_losses
ylayer_metrics

zlayers
{metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_7/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_7/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_28/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_28/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_29/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_29/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_30/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_30/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_31/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_31/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_14/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_14/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_7/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_7/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE

20
31
 
 

0
1

|0
}1
 
 
 
░
9trainable_variables
~non_trainable_variables
:	variables
;regularization_losses
layer_regularization_losses
Аlayer_metrics
Бlayers
Вmetrics
 

&0
'1

&0
'1
22
33
 
▓
>trainable_variables
Гnon_trainable_variables
?	variables
@regularization_losses
 Дlayer_regularization_losses
Еlayer_metrics
Жlayers
Зmetrics

(0
)1

(0
)1
 
▓
Btrainable_variables
Иnon_trainable_variables
C	variables
Dregularization_losses
 Йlayer_regularization_losses
Кlayer_metrics
Лlayers
Мmetrics
 
 
 
▓
Ftrainable_variables
Нnon_trainable_variables
G	variables
Hregularization_losses
 Оlayer_regularization_losses
Пlayer_metrics
Рlayers
Сmetrics

*0
+1

*0
+1
 
▓
Jtrainable_variables
Тnon_trainable_variables
K	variables
Lregularization_losses
 Уlayer_regularization_losses
Фlayer_metrics
Хlayers
Цmetrics
 
 
 
▓
Ntrainable_variables
Чnon_trainable_variables
O	variables
Pregularization_losses
 Шlayer_regularization_losses
Щlayer_metrics
Ъlayers
Ыmetrics

,0
-1

,0
-1
 
▓
Rtrainable_variables
Ьnon_trainable_variables
S	variables
Tregularization_losses
 Эlayer_regularization_losses
Юlayer_metrics
Яlayers
аmetrics
 
 
 
▓
Vtrainable_variables
бnon_trainable_variables
W	variables
Xregularization_losses
 вlayer_regularization_losses
гlayer_metrics
дlayers
еmetrics

.0
/1

.0
/1
 
▓
Ztrainable_variables
жnon_trainable_variables
[	variables
\regularization_losses
 зlayer_regularization_losses
иlayer_metrics
йlayers
кmetrics
 
 
 
▓
^trainable_variables
лnon_trainable_variables
_	variables
`regularization_losses
 мlayer_regularization_losses
нlayer_metrics
оlayers
пmetrics
 
 
 
▓
btrainable_variables
░non_trainable_variables
c	variables
dregularization_losses
 ▒layer_regularization_losses
▓layer_metrics
│layers
┤metrics
 
 
 
▓
ftrainable_variables
╡non_trainable_variables
g	variables
hregularization_losses
 ╢layer_regularization_losses
╖layer_metrics
╕layers
╣metrics

00
11

00
11
 
▓
jtrainable_variables
║non_trainable_variables
k	variables
lregularization_losses
 ╗layer_regularization_losses
╝layer_metrics
╜layers
╛metrics
 
 
 
▓
ntrainable_variables
┐non_trainable_variables
o	variables
pregularization_losses
 └layer_regularization_losses
┴layer_metrics
┬layers
├metrics

20
31
 
 
f
	0

1
2
3
4
5
6
7
8
9
10
11
12
13
 
 
 
 
 
 
8

─total

┼count
╞	variables
╟	keras_api
I

╚total

╔count
╩
_fn_kwargs
╦	variables
╠	keras_api
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

─0
┼1

╞	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

╚0
╔1

╦	variables
qo
VARIABLE_VALUEAdam/dense_15/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_15/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_28/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_28/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_29/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_29/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_30/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_30/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_31/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_31/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_14/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_14/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_15/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_15/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_28/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_28/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_29/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_29/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_30/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_30/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_31/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_31/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_14/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_14/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
в
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*
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
*2
config_proto" 

CPU

GPU2 *0J 8В *-
f(R&
$__inference_signature_wrapper_833100
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
я
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp$conv2d_30/kernel/Read/ReadVariableOp"conv2d_30/bias/Read/ReadVariableOp$conv2d_31/kernel/Read/ReadVariableOp"conv2d_31/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp+Adam/conv2d_28/kernel/m/Read/ReadVariableOp)Adam/conv2d_28/bias/m/Read/ReadVariableOp+Adam/conv2d_29/kernel/m/Read/ReadVariableOp)Adam/conv2d_29/bias/m/Read/ReadVariableOp+Adam/conv2d_30/kernel/m/Read/ReadVariableOp)Adam/conv2d_30/bias/m/Read/ReadVariableOp+Adam/conv2d_31/kernel/m/Read/ReadVariableOp)Adam/conv2d_31/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp+Adam/conv2d_28/kernel/v/Read/ReadVariableOp)Adam/conv2d_28/bias/v/Read/ReadVariableOp+Adam/conv2d_29/kernel/v/Read/ReadVariableOp)Adam/conv2d_29/bias/v/Read/ReadVariableOp+Adam/conv2d_30/kernel/v/Read/ReadVariableOp)Adam/conv2d_30/bias/v/Read/ReadVariableOp+Adam/conv2d_31/kernel/v/Read/ReadVariableOp)Adam/conv2d_31/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOpConst*B
Tin;
927	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *(
f#R!
__inference__traced_save_834655
╞
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_15/kerneldense_15/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_7/gammabatch_normalization_7/betaconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasdense_14/kerneldense_14/bias!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancetotalcounttotal_1count_1Adam/dense_15/kernel/mAdam/dense_15/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/conv2d_28/kernel/mAdam/conv2d_28/bias/mAdam/conv2d_29/kernel/mAdam/conv2d_29/bias/mAdam/conv2d_30/kernel/mAdam/conv2d_30/bias/mAdam/conv2d_31/kernel/mAdam/conv2d_31/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_15/kernel/vAdam/dense_15/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/conv2d_28/kernel/vAdam/conv2d_28/bias/vAdam/conv2d_29/kernel/vAdam/conv2d_29/bias/vAdam/conv2d_30/kernel/vAdam/conv2d_30/bias/vAdam/conv2d_31/kernel/vAdam/conv2d_31/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/v*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__traced_restore_834824│у
╣
м
D__inference_dense_14_layer_call_and_return_conditional_losses_834415

inputs2
matmul_readvariableop_resource:
А@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╟
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
ў
d
F__inference_dropout_15_layer_call_and_return_conditional_losses_834429

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╒
╘
!__inference__wrapped_model_831848
input_1
cnn3_831814:
cnn3_831816:
cnn3_831818:
cnn3_831820:%
cnn3_831822: 
cnn3_831824: &
cnn3_831826: А
cnn3_831828:	А'
cnn3_831830:АА
cnn3_831832:	А'
cnn3_831834:АА
cnn3_831836:	А
cnn3_831838:
А@А
cnn3_831840:	А
cnn3_831842:	А
cnn3_831844:
identityИвCNN3/StatefulPartitionedCallп
CNN3/StatefulPartitionedCallStatefulPartitionedCallinput_1cnn3_831814cnn3_831816cnn3_831818cnn3_831820cnn3_831822cnn3_831824cnn3_831826cnn3_831828cnn3_831830cnn3_831832cnn3_831834cnn3_831836cnn3_831838cnn3_831840cnn3_831842cnn3_831844*
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
*2
config_proto" 

CPU

GPU2 *0J 8В * 
fR
__inference_call_7964712
CNN3/StatefulPartitionedCallШ
IdentityIdentity%CNN3/StatefulPartitionedCall:output:0^CNN3/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2<
CNN3/StatefulPartitionedCallCNN3/StatefulPartitionedCall:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
Е(
■
@__inference_CNN3_layer_call_and_return_conditional_losses_832737

inputs!
sequential_7_832678:!
sequential_7_832680:!
sequential_7_832682:!
sequential_7_832684:-
sequential_7_832686: !
sequential_7_832688: .
sequential_7_832690: А"
sequential_7_832692:	А/
sequential_7_832694:АА"
sequential_7_832696:	А/
sequential_7_832698:АА"
sequential_7_832700:	А'
sequential_7_832702:
А@А"
sequential_7_832704:	А"
dense_15_832719:	А
dense_15_832721:
identityИв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpв dense_15/StatefulPartitionedCallв$sequential_7/StatefulPartitionedCall┬
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7_832678sequential_7_832680sequential_7_832682sequential_7_832684sequential_7_832686sequential_7_832688sequential_7_832690sequential_7_832692sequential_7_832694sequential_7_832696sequential_7_832698sequential_7_832700sequential_7_832702sequential_7_832704*
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
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_8322022&
$sequential_7/StatefulPartitionedCall└
 dense_15/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0dense_15_832719dense_15_832721*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_8327182"
 dense_15/StatefulPartitionedCall─
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_832686*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╝
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_832702* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul░
IdentityIdentity)dense_15/StatefulPartitionedCall:output:03^conv2d_28/kernel/Regularizer/Square/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp!^dense_15/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ь
╤
6__inference_batch_normalization_7_layer_call_fn_834236

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall╕
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
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8319142
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ю
╤
6__inference_batch_normalization_7_layer_call_fn_834223

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCall║
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
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8318702
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Фw
■
__inference_call_798486

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_28_conv2d_readvariableop_resource: D
6sequential_7_conv2d_28_biasadd_readvariableop_resource: P
5sequential_7_conv2d_29_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_29_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_30_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_30_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_31_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_31_biasadd_readvariableop_resource:	АH
4sequential_7_dense_14_matmul_readvariableop_resource:
А@АD
5sequential_7_dense_14_biasadd_readvariableop_resource:	А:
'dense_15_matmul_readvariableop_resource:	А6
(dense_15_biasadd_readvariableop_resource:
identityИвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_28/BiasAdd/ReadVariableOpв,sequential_7/conv2d_28/Conv2D/ReadVariableOpв-sequential_7/conv2d_29/BiasAdd/ReadVariableOpв,sequential_7/conv2d_29/Conv2D/ReadVariableOpв-sequential_7/conv2d_30/BiasAdd/ReadVariableOpв,sequential_7/conv2d_30/Conv2D/ReadVariableOpв-sequential_7/conv2d_31/BiasAdd/ReadVariableOpв,sequential_7/conv2d_31/Conv2D/ReadVariableOpв,sequential_7/dense_14/BiasAdd/ReadVariableOpв+sequential_7/dense_14/MatMul/ReadVariableOpп
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2ы
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_28/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_28/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_28/Conv2D╤
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpф
sequential_7/conv2d_28/BiasAddBiasAdd&sequential_7/conv2d_28/Conv2D:output:05sequential_7/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_28/BiasAddе
sequential_7/conv2d_28/ReluRelu'sequential_7/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_28/Reluё
%sequential_7/max_pooling2d_28/MaxPoolMaxPool)sequential_7/conv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_28/MaxPool█
,sequential_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_29/Conv2D/ReadVariableOpС
sequential_7/conv2d_29/Conv2DConv2D.sequential_7/max_pooling2d_28/MaxPool:output:04sequential_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_7/conv2d_29/Conv2D╥
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpх
sequential_7/conv2d_29/BiasAddBiasAdd&sequential_7/conv2d_29/Conv2D:output:05sequential_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_7/conv2d_29/BiasAddж
sequential_7/conv2d_29/ReluRelu'sequential_7/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_7/conv2d_29/ReluЄ
%sequential_7/max_pooling2d_29/MaxPoolMaxPool)sequential_7/conv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_29/MaxPool▄
,sequential_7/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_30/Conv2D/ReadVariableOpС
sequential_7/conv2d_30/Conv2DConv2D.sequential_7/max_pooling2d_29/MaxPool:output:04sequential_7/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_30/Conv2D╥
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpх
sequential_7/conv2d_30/BiasAddBiasAdd&sequential_7/conv2d_30/Conv2D:output:05sequential_7/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_30/BiasAddж
sequential_7/conv2d_30/ReluRelu'sequential_7/conv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_30/ReluЄ
%sequential_7/max_pooling2d_30/MaxPoolMaxPool)sequential_7/conv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_30/MaxPool▄
,sequential_7/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_31/Conv2D/ReadVariableOpС
sequential_7/conv2d_31/Conv2DConv2D.sequential_7/max_pooling2d_30/MaxPool:output:04sequential_7/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_7/conv2d_31/Conv2D╥
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpх
sequential_7/conv2d_31/BiasAddBiasAdd&sequential_7/conv2d_31/Conv2D:output:05sequential_7/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_7/conv2d_31/BiasAddж
sequential_7/conv2d_31/ReluRelu'sequential_7/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_7/conv2d_31/ReluЄ
%sequential_7/max_pooling2d_31/MaxPoolMaxPool)sequential_7/conv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_31/MaxPool╗
 sequential_7/dropout_14/IdentityIdentity.sequential_7/max_pooling2d_31/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_7/dropout_14/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_14/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_7/dense_14/MatMul/ReadVariableOp╫
sequential_7/dense_14/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/MatMul╧
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOp┌
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/BiasAddЫ
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/Reluн
 sequential_7/dropout_15/IdentityIdentity(sequential_7/dense_14/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_15/Identityй
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_15/MatMul/ReadVariableOp▒
dense_15/MatMulMatMul)sequential_7/dropout_15/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/MatMulз
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpе
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/BiasAdd|
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_15/SoftmaxА
IdentityIdentitydense_15/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_28/BiasAdd/ReadVariableOp-^sequential_7/conv2d_28/Conv2D/ReadVariableOp.^sequential_7/conv2d_29/BiasAdd/ReadVariableOp-^sequential_7/conv2d_29/Conv2D/ReadVariableOp.^sequential_7/conv2d_30/BiasAdd/ReadVariableOp-^sequential_7/conv2d_30/Conv2D/ReadVariableOp.^sequential_7/conv2d_31/BiasAdd/ReadVariableOp-^sequential_7/conv2d_31/Conv2D/ReadVariableOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp-sequential_7/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_28/Conv2D/ReadVariableOp,sequential_7/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp-sequential_7/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_29/Conv2D/ReadVariableOp,sequential_7/conv2d_29/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp-sequential_7/conv2d_30/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_30/Conv2D/ReadVariableOp,sequential_7/conv2d_30/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp-sequential_7/conv2d_31/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_31/Conv2D/ReadVariableOp,sequential_7/conv2d_31/Conv2D/ReadVariableOp2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╡
e
F__inference_dropout_15_layer_call_and_return_conditional_losses_832253

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
├
Ь
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_832056

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Л
Ь
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_831870

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
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
┐
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834174

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
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
ШМ
С
@__inference_CNN3_layer_call_and_return_conditional_losses_833366
input_1H
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_28_conv2d_readvariableop_resource: D
6sequential_7_conv2d_28_biasadd_readvariableop_resource: P
5sequential_7_conv2d_29_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_29_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_30_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_30_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_31_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_31_biasadd_readvariableop_resource:	АH
4sequential_7_dense_14_matmul_readvariableop_resource:
А@АD
5sequential_7_dense_14_biasadd_readvariableop_resource:	А:
'dense_15_matmul_readvariableop_resource:	А6
(dense_15_biasadd_readvariableop_resource:
identityИв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_28/BiasAdd/ReadVariableOpв,sequential_7/conv2d_28/Conv2D/ReadVariableOpв-sequential_7/conv2d_29/BiasAdd/ReadVariableOpв,sequential_7/conv2d_29/Conv2D/ReadVariableOpв-sequential_7/conv2d_30/BiasAdd/ReadVariableOpв,sequential_7/conv2d_30/Conv2D/ReadVariableOpв-sequential_7/conv2d_31/BiasAdd/ReadVariableOpв,sequential_7/conv2d_31/Conv2D/ReadVariableOpв,sequential_7/dense_14/BiasAdd/ReadVariableOpв+sequential_7/dense_14/MatMul/ReadVariableOpп
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2ь
#sequential_7/lambda_7/strided_sliceStridedSliceinput_12sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_28/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_28/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_28/Conv2D╤
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpф
sequential_7/conv2d_28/BiasAddBiasAdd&sequential_7/conv2d_28/Conv2D:output:05sequential_7/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_28/BiasAddе
sequential_7/conv2d_28/ReluRelu'sequential_7/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_28/Reluё
%sequential_7/max_pooling2d_28/MaxPoolMaxPool)sequential_7/conv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_28/MaxPool█
,sequential_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_29/Conv2D/ReadVariableOpС
sequential_7/conv2d_29/Conv2DConv2D.sequential_7/max_pooling2d_28/MaxPool:output:04sequential_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_7/conv2d_29/Conv2D╥
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpх
sequential_7/conv2d_29/BiasAddBiasAdd&sequential_7/conv2d_29/Conv2D:output:05sequential_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_7/conv2d_29/BiasAddж
sequential_7/conv2d_29/ReluRelu'sequential_7/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_7/conv2d_29/ReluЄ
%sequential_7/max_pooling2d_29/MaxPoolMaxPool)sequential_7/conv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_29/MaxPool▄
,sequential_7/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_30/Conv2D/ReadVariableOpС
sequential_7/conv2d_30/Conv2DConv2D.sequential_7/max_pooling2d_29/MaxPool:output:04sequential_7/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_30/Conv2D╥
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpх
sequential_7/conv2d_30/BiasAddBiasAdd&sequential_7/conv2d_30/Conv2D:output:05sequential_7/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_30/BiasAddж
sequential_7/conv2d_30/ReluRelu'sequential_7/conv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_30/ReluЄ
%sequential_7/max_pooling2d_30/MaxPoolMaxPool)sequential_7/conv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_30/MaxPool▄
,sequential_7/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_31/Conv2D/ReadVariableOpС
sequential_7/conv2d_31/Conv2DConv2D.sequential_7/max_pooling2d_30/MaxPool:output:04sequential_7/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_7/conv2d_31/Conv2D╥
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpх
sequential_7/conv2d_31/BiasAddBiasAdd&sequential_7/conv2d_31/Conv2D:output:05sequential_7/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_7/conv2d_31/BiasAddж
sequential_7/conv2d_31/ReluRelu'sequential_7/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_7/conv2d_31/ReluЄ
%sequential_7/max_pooling2d_31/MaxPoolMaxPool)sequential_7/conv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_31/MaxPool╗
 sequential_7/dropout_14/IdentityIdentity.sequential_7/max_pooling2d_31/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_7/dropout_14/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_14/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_7/dense_14/MatMul/ReadVariableOp╫
sequential_7/dense_14/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/MatMul╧
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOp┌
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/BiasAddЫ
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/Reluн
 sequential_7/dropout_15/IdentityIdentity(sequential_7/dense_14/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_15/Identityй
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_15/MatMul/ReadVariableOp▒
dense_15/MatMulMatMul)sequential_7/dropout_15/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/MatMulз
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpе
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/BiasAdd|
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_15/Softmaxц
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul▌
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulщ
IdentityIdentitydense_15/Softmax:softmax:03^conv2d_28/kernel/Regularizer/Square/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_28/BiasAdd/ReadVariableOp-^sequential_7/conv2d_28/Conv2D/ReadVariableOp.^sequential_7/conv2d_29/BiasAdd/ReadVariableOp-^sequential_7/conv2d_29/Conv2D/ReadVariableOp.^sequential_7/conv2d_30/BiasAdd/ReadVariableOp-^sequential_7/conv2d_30/Conv2D/ReadVariableOp.^sequential_7/conv2d_31/BiasAdd/ReadVariableOp-^sequential_7/conv2d_31/Conv2D/ReadVariableOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp-sequential_7/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_28/Conv2D/ReadVariableOp,sequential_7/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp-sequential_7/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_29/Conv2D/ReadVariableOp,sequential_7/conv2d_29/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp-sequential_7/conv2d_30/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_30/Conv2D/ReadVariableOp,sequential_7/conv2d_30/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp-sequential_7/conv2d_31/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_31/Conv2D/ReadVariableOp,sequential_7/conv2d_31/Conv2D/ReadVariableOp2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
ў
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_832368

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╕

Ў
D__inference_dense_15_layer_call_and_return_conditional_losses_834103

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
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
г
Ч
)__inference_dense_15_layer_call_fn_834112

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall∙
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
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_8327182
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Л
Ь
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834156

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
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
ХМ
Р
@__inference_CNN3_layer_call_and_return_conditional_losses_833184

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_28_conv2d_readvariableop_resource: D
6sequential_7_conv2d_28_biasadd_readvariableop_resource: P
5sequential_7_conv2d_29_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_29_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_30_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_30_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_31_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_31_biasadd_readvariableop_resource:	АH
4sequential_7_dense_14_matmul_readvariableop_resource:
А@АD
5sequential_7_dense_14_biasadd_readvariableop_resource:	А:
'dense_15_matmul_readvariableop_resource:	А6
(dense_15_biasadd_readvariableop_resource:
identityИв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_28/BiasAdd/ReadVariableOpв,sequential_7/conv2d_28/Conv2D/ReadVariableOpв-sequential_7/conv2d_29/BiasAdd/ReadVariableOpв,sequential_7/conv2d_29/Conv2D/ReadVariableOpв-sequential_7/conv2d_30/BiasAdd/ReadVariableOpв,sequential_7/conv2d_30/Conv2D/ReadVariableOpв-sequential_7/conv2d_31/BiasAdd/ReadVariableOpв,sequential_7/conv2d_31/Conv2D/ReadVariableOpв,sequential_7/dense_14/BiasAdd/ReadVariableOpв+sequential_7/dense_14/MatMul/ReadVariableOpп
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2ы
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_28/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_28/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_28/Conv2D╤
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpф
sequential_7/conv2d_28/BiasAddBiasAdd&sequential_7/conv2d_28/Conv2D:output:05sequential_7/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_28/BiasAddе
sequential_7/conv2d_28/ReluRelu'sequential_7/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_28/Reluё
%sequential_7/max_pooling2d_28/MaxPoolMaxPool)sequential_7/conv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_28/MaxPool█
,sequential_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_29/Conv2D/ReadVariableOpС
sequential_7/conv2d_29/Conv2DConv2D.sequential_7/max_pooling2d_28/MaxPool:output:04sequential_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_7/conv2d_29/Conv2D╥
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpх
sequential_7/conv2d_29/BiasAddBiasAdd&sequential_7/conv2d_29/Conv2D:output:05sequential_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_7/conv2d_29/BiasAddж
sequential_7/conv2d_29/ReluRelu'sequential_7/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_7/conv2d_29/ReluЄ
%sequential_7/max_pooling2d_29/MaxPoolMaxPool)sequential_7/conv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_29/MaxPool▄
,sequential_7/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_30/Conv2D/ReadVariableOpС
sequential_7/conv2d_30/Conv2DConv2D.sequential_7/max_pooling2d_29/MaxPool:output:04sequential_7/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_30/Conv2D╥
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpх
sequential_7/conv2d_30/BiasAddBiasAdd&sequential_7/conv2d_30/Conv2D:output:05sequential_7/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_30/BiasAddж
sequential_7/conv2d_30/ReluRelu'sequential_7/conv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_30/ReluЄ
%sequential_7/max_pooling2d_30/MaxPoolMaxPool)sequential_7/conv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_30/MaxPool▄
,sequential_7/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_31/Conv2D/ReadVariableOpС
sequential_7/conv2d_31/Conv2DConv2D.sequential_7/max_pooling2d_30/MaxPool:output:04sequential_7/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_7/conv2d_31/Conv2D╥
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpх
sequential_7/conv2d_31/BiasAddBiasAdd&sequential_7/conv2d_31/Conv2D:output:05sequential_7/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_7/conv2d_31/BiasAddж
sequential_7/conv2d_31/ReluRelu'sequential_7/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_7/conv2d_31/ReluЄ
%sequential_7/max_pooling2d_31/MaxPoolMaxPool)sequential_7/conv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_31/MaxPool╗
 sequential_7/dropout_14/IdentityIdentity.sequential_7/max_pooling2d_31/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_7/dropout_14/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_14/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_7/dense_14/MatMul/ReadVariableOp╫
sequential_7/dense_14/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/MatMul╧
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOp┌
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/BiasAddЫ
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/Reluн
 sequential_7/dropout_15/IdentityIdentity(sequential_7/dense_14/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_15/Identityй
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_15/MatMul/ReadVariableOp▒
dense_15/MatMulMatMul)sequential_7/dropout_15/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/MatMulз
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpе
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/BiasAdd|
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_15/Softmaxц
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul▌
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulщ
IdentityIdentitydense_15/Softmax:softmax:03^conv2d_28/kernel/Regularizer/Square/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_28/BiasAdd/ReadVariableOp-^sequential_7/conv2d_28/Conv2D/ReadVariableOp.^sequential_7/conv2d_29/BiasAdd/ReadVariableOp-^sequential_7/conv2d_29/Conv2D/ReadVariableOp.^sequential_7/conv2d_30/BiasAdd/ReadVariableOp-^sequential_7/conv2d_30/Conv2D/ReadVariableOp.^sequential_7/conv2d_31/BiasAdd/ReadVariableOp-^sequential_7/conv2d_31/Conv2D/ReadVariableOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp-sequential_7/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_28/Conv2D/ReadVariableOp,sequential_7/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp-sequential_7/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_29/Conv2D/ReadVariableOp,sequential_7/conv2d_29/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp-sequential_7/conv2d_30/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_30/Conv2D/ReadVariableOp,sequential_7/conv2d_30/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp-sequential_7/conv2d_31/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_31/Conv2D/ReadVariableOp,sequential_7/conv2d_31/Conv2D/ReadVariableOp2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╕

Ў
D__inference_dense_15_layer_call_and_return_conditional_losses_832718

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
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
▐
M
1__inference_max_pooling2d_30_layer_call_fn_832010

inputs
identityЄ
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
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_8320042
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
м
h
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_832004

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╡
e
F__inference_dropout_15_layer_call_and_return_conditional_losses_834441

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
м
h
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_831980

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┐
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_831914

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
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
╦R
р
H__inference_sequential_7_layer_call_and_return_conditional_losses_832202

inputs*
batch_normalization_7_832057:*
batch_normalization_7_832059:*
batch_normalization_7_832061:*
batch_normalization_7_832063:*
conv2d_28_832084: 
conv2d_28_832086: +
conv2d_29_832102: А
conv2d_29_832104:	А,
conv2d_30_832120:АА
conv2d_30_832122:	А,
conv2d_31_832138:АА
conv2d_31_832140:	А#
dense_14_832177:
А@А
dense_14_832179:	А
identityИв-batch_normalization_7/StatefulPartitionedCallв!conv2d_28/StatefulPartitionedCallв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв!conv2d_29/StatefulPartitionedCallв!conv2d_30/StatefulPartitionedCallв!conv2d_31/StatefulPartitionedCallв dense_14/StatefulPartitionedCallв1dense_14/kernel/Regularizer/Square/ReadVariableOpс
lambda_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_lambda_7_layer_call_and_return_conditional_losses_8320372
lambda_7/PartitionedCall╜
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0batch_normalization_7_832057batch_normalization_7_832059batch_normalization_7_832061batch_normalization_7_832063*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8320562/
-batch_normalization_7/StatefulPartitionedCall╓
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_28_832084conv2d_28_832086*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_8320832#
!conv2d_28/StatefulPartitionedCallЭ
 max_pooling2d_28/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         %% * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_8319802"
 max_pooling2d_28/PartitionedCall╩
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_28/PartitionedCall:output:0conv2d_29_832102conv2d_29_832104*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_8321012#
!conv2d_29/StatefulPartitionedCallЮ
 max_pooling2d_29/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_8319922"
 max_pooling2d_29/PartitionedCall╩
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_29/PartitionedCall:output:0conv2d_30_832120conv2d_30_832122*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_30_layer_call_and_return_conditional_losses_8321192#
!conv2d_30/StatefulPartitionedCallЮ
 max_pooling2d_30/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_8320042"
 max_pooling2d_30/PartitionedCall╩
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_30/PartitionedCall:output:0conv2d_31_832138conv2d_31_832140*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_31_layer_call_and_return_conditional_losses_8321372#
!conv2d_31/StatefulPartitionedCallЮ
 max_pooling2d_31/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_8320162"
 max_pooling2d_31/PartitionedCallЛ
dropout_14/PartitionedCallPartitionedCall)max_pooling2d_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_8321492
dropout_14/PartitionedCall·
flatten_7/PartitionedCallPartitionedCall#dropout_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_8321572
flatten_7/PartitionedCall╢
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_14_832177dense_14_832179*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_8321762"
 dense_14/StatefulPartitionedCallГ
dropout_15/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_8321872
dropout_15/PartitionedCall┴
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_832084*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╕
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_14_832177* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul─
IdentityIdentity#dropout_15/PartitionedCall:output:0.^batch_normalization_7/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ч
d
F__inference_dropout_14_layer_call_and_return_conditional_losses_834359

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╩
Я
*__inference_conv2d_28_layer_call_fn_834294

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallВ
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
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_8320832
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Э
А
E__inference_conv2d_29_layer_call_and_return_conditional_losses_834305

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%А2

Identity"
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
┼m
¤
H__inference_sequential_7_layer_call_and_return_conditional_losses_833869
lambda_7_input;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_28_conv2d_readvariableop_resource: 7
)conv2d_28_biasadd_readvariableop_resource: C
(conv2d_29_conv2d_readvariableop_resource: А8
)conv2d_29_biasadd_readvariableop_resource:	АD
(conv2d_30_conv2d_readvariableop_resource:АА8
)conv2d_30_biasadd_readvariableop_resource:	АD
(conv2d_31_conv2d_readvariableop_resource:АА8
)conv2d_31_biasadd_readvariableop_resource:	А;
'dense_14_matmul_readvariableop_resource:
А@А7
(dense_14_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1в conv2d_28/BiasAdd/ReadVariableOpвconv2d_28/Conv2D/ReadVariableOpв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв conv2d_29/BiasAdd/ReadVariableOpвconv2d_29/Conv2D/ReadVariableOpв conv2d_30/BiasAdd/ReadVariableOpвconv2d_30/Conv2D/ReadVariableOpв conv2d_31/BiasAdd/ReadVariableOpвconv2d_31/Conv2D/ReadVariableOpвdense_14/BiasAdd/ReadVariableOpвdense_14/MatMul/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpХ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЩ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Щ
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2▓
lambda_7/strided_sliceStridedSlicelambda_7_input%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_7/strided_slice╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3│
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_28/Conv2D/ReadVariableOpх
conv2d_28/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_28/Conv2Dк
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp░
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_28/BiasAdd~
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_28/Relu╩
max_pooling2d_28/MaxPoolMaxPoolconv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_28/MaxPool┤
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_29/Conv2D/ReadVariableOp▌
conv2d_29/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_29/Conv2Dл
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp▒
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_29/BiasAdd
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_29/Relu╦
max_pooling2d_29/MaxPoolMaxPoolconv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_29/MaxPool╡
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_30/Conv2D/ReadVariableOp▌
conv2d_30/Conv2DConv2D!max_pooling2d_29/MaxPool:output:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_30/Conv2Dл
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp▒
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_30/BiasAdd
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_30/Relu╦
max_pooling2d_30/MaxPoolMaxPoolconv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_30/MaxPool╡
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_31/Conv2D/ReadVariableOp▌
conv2d_31/Conv2DConv2D!max_pooling2d_30/MaxPool:output:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
conv2d_31/Conv2Dл
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp▒
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
conv2d_31/BiasAdd
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
conv2d_31/Relu╦
max_pooling2d_31/MaxPoolMaxPoolconv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_31/MaxPoolФ
dropout_14/IdentityIdentity!max_pooling2d_31/MaxPool:output:0*
T0*0
_output_shapes
:         А2
dropout_14/Identitys
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_7/ConstЬ
flatten_7/ReshapeReshapedropout_14/Identity:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_7/Reshapeк
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_14/MatMul/ReadVariableOpг
dense_14/MatMulMatMulflatten_7/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_14/MatMulи
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_14/BiasAdd/ReadVariableOpж
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_14/BiasAddt
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_14/ReluЖ
dropout_15/IdentityIdentitydense_14/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_15/Identity┘
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╨
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulє
IdentityIdentitydropout_15/Identity:output:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_7_input
ЪИ
┼
H__inference_sequential_7_layer_call_and_return_conditional_losses_833792

inputs;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_28_conv2d_readvariableop_resource: 7
)conv2d_28_biasadd_readvariableop_resource: C
(conv2d_29_conv2d_readvariableop_resource: А8
)conv2d_29_biasadd_readvariableop_resource:	АD
(conv2d_30_conv2d_readvariableop_resource:АА8
)conv2d_30_biasadd_readvariableop_resource:	АD
(conv2d_31_conv2d_readvariableop_resource:АА8
)conv2d_31_biasadd_readvariableop_resource:	А;
'dense_14_matmul_readvariableop_resource:
А@А7
(dense_14_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_7/AssignNewValueв&batch_normalization_7/AssignNewValue_1в5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1в conv2d_28/BiasAdd/ReadVariableOpвconv2d_28/Conv2D/ReadVariableOpв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв conv2d_29/BiasAdd/ReadVariableOpвconv2d_29/Conv2D/ReadVariableOpв conv2d_30/BiasAdd/ReadVariableOpвconv2d_30/Conv2D/ReadVariableOpв conv2d_31/BiasAdd/ReadVariableOpвconv2d_31/Conv2D/ReadVariableOpвdense_14/BiasAdd/ReadVariableOpвdense_14/MatMul/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpХ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЩ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Щ
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2к
lambda_7/strided_sliceStridedSliceinputs%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_7/strided_slice╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ї
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_7/FusedBatchNormV3░
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue╝
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1│
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_28/Conv2D/ReadVariableOpх
conv2d_28/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_28/Conv2Dк
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp░
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_28/BiasAdd~
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_28/Relu╩
max_pooling2d_28/MaxPoolMaxPoolconv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_28/MaxPool┤
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_29/Conv2D/ReadVariableOp▌
conv2d_29/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_29/Conv2Dл
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp▒
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_29/BiasAdd
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_29/Relu╦
max_pooling2d_29/MaxPoolMaxPoolconv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_29/MaxPool╡
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_30/Conv2D/ReadVariableOp▌
conv2d_30/Conv2DConv2D!max_pooling2d_29/MaxPool:output:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_30/Conv2Dл
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp▒
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_30/BiasAdd
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_30/Relu╦
max_pooling2d_30/MaxPoolMaxPoolconv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_30/MaxPool╡
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_31/Conv2D/ReadVariableOp▌
conv2d_31/Conv2DConv2D!max_pooling2d_30/MaxPool:output:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
conv2d_31/Conv2Dл
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp▒
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
conv2d_31/BiasAdd
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
conv2d_31/Relu╦
max_pooling2d_31/MaxPoolMaxPoolconv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_31/MaxPooly
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_14/dropout/Const╕
dropout_14/dropout/MulMul!max_pooling2d_31/MaxPool:output:0!dropout_14/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout_14/dropout/MulЕ
dropout_14/dropout/ShapeShape!max_pooling2d_31/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shape▐
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype021
/dropout_14/dropout/random_uniform/RandomUniformЛ
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_14/dropout/GreaterEqual/yє
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2!
dropout_14/dropout/GreaterEqualй
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout_14/dropout/Castп
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout_14/dropout/Mul_1s
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_7/ConstЬ
flatten_7/ReshapeReshapedropout_14/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_7/Reshapeк
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_14/MatMul/ReadVariableOpг
dense_14/MatMulMatMulflatten_7/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_14/MatMulи
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_14/BiasAdd/ReadVariableOpж
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_14/BiasAddt
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_14/Reluy
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_15/dropout/Constк
dropout_15/dropout/MulMuldense_14/Relu:activations:0!dropout_15/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_15/dropout/Mul
dropout_15/dropout/ShapeShapedense_14/Relu:activations:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shape╓
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_15/dropout/random_uniform/RandomUniformЛ
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_15/dropout/GreaterEqual/yы
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_15/dropout/GreaterEqualб
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_15/dropout/Castз
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_15/dropout/Mul_1┘
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╨
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul├
IdentityIdentitydropout_15/dropout/Mul_1:z:0%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▐
M
1__inference_max_pooling2d_28_layer_call_fn_831986

inputs
identityЄ
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
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_8319802
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
с
E
)__inference_lambda_7_layer_call_fn_834133

inputs
identity╧
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
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_lambda_7_layer_call_and_return_conditional_losses_8320372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
м
h
L__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_831992

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ў
└
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834210

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┘
Д
-__inference_sequential_7_layer_call_fn_834059

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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А
identityИвStatefulPartitionedCallЫ
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
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_8324912
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
б
Б
E__inference_conv2d_31_layer_call_and_return_conditional_losses_832137

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
Ч
d
F__inference_dropout_14_layer_call_and_return_conditional_losses_832149

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ж
╤
6__inference_batch_normalization_7_layer_call_fn_834249

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8320562
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▓И
═
H__inference_sequential_7_layer_call_and_return_conditional_losses_833960
lambda_7_input;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_28_conv2d_readvariableop_resource: 7
)conv2d_28_biasadd_readvariableop_resource: C
(conv2d_29_conv2d_readvariableop_resource: А8
)conv2d_29_biasadd_readvariableop_resource:	АD
(conv2d_30_conv2d_readvariableop_resource:АА8
)conv2d_30_biasadd_readvariableop_resource:	АD
(conv2d_31_conv2d_readvariableop_resource:АА8
)conv2d_31_biasadd_readvariableop_resource:	А;
'dense_14_matmul_readvariableop_resource:
А@А7
(dense_14_biasadd_readvariableop_resource:	А
identityИв$batch_normalization_7/AssignNewValueв&batch_normalization_7/AssignNewValue_1в5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1в conv2d_28/BiasAdd/ReadVariableOpвconv2d_28/Conv2D/ReadVariableOpв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв conv2d_29/BiasAdd/ReadVariableOpвconv2d_29/Conv2D/ReadVariableOpв conv2d_30/BiasAdd/ReadVariableOpвconv2d_30/Conv2D/ReadVariableOpв conv2d_31/BiasAdd/ReadVariableOpвconv2d_31/Conv2D/ReadVariableOpвdense_14/BiasAdd/ReadVariableOpвdense_14/MatMul/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpХ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЩ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Щ
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2▓
lambda_7/strided_sliceStridedSlicelambda_7_input%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_7/strided_slice╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ї
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_7/FusedBatchNormV3░
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue╝
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1│
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_28/Conv2D/ReadVariableOpх
conv2d_28/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_28/Conv2Dк
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp░
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_28/BiasAdd~
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_28/Relu╩
max_pooling2d_28/MaxPoolMaxPoolconv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_28/MaxPool┤
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_29/Conv2D/ReadVariableOp▌
conv2d_29/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_29/Conv2Dл
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp▒
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_29/BiasAdd
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_29/Relu╦
max_pooling2d_29/MaxPoolMaxPoolconv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_29/MaxPool╡
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_30/Conv2D/ReadVariableOp▌
conv2d_30/Conv2DConv2D!max_pooling2d_29/MaxPool:output:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_30/Conv2Dл
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp▒
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_30/BiasAdd
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_30/Relu╦
max_pooling2d_30/MaxPoolMaxPoolconv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_30/MaxPool╡
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_31/Conv2D/ReadVariableOp▌
conv2d_31/Conv2DConv2D!max_pooling2d_30/MaxPool:output:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
conv2d_31/Conv2Dл
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp▒
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
conv2d_31/BiasAdd
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
conv2d_31/Relu╦
max_pooling2d_31/MaxPoolMaxPoolconv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_31/MaxPooly
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_14/dropout/Const╕
dropout_14/dropout/MulMul!max_pooling2d_31/MaxPool:output:0!dropout_14/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout_14/dropout/MulЕ
dropout_14/dropout/ShapeShape!max_pooling2d_31/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shape▐
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype021
/dropout_14/dropout/random_uniform/RandomUniformЛ
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_14/dropout/GreaterEqual/yє
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2!
dropout_14/dropout/GreaterEqualй
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout_14/dropout/Castп
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout_14/dropout/Mul_1s
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_7/ConstЬ
flatten_7/ReshapeReshapedropout_14/dropout/Mul_1:z:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_7/Reshapeк
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_14/MatMul/ReadVariableOpг
dense_14/MatMulMatMulflatten_7/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_14/MatMulи
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_14/BiasAdd/ReadVariableOpж
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_14/BiasAddt
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_14/Reluy
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_15/dropout/Constк
dropout_15/dropout/MulMuldense_14/Relu:activations:0!dropout_15/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_15/dropout/Mul
dropout_15/dropout/ShapeShapedense_14/Relu:activations:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shape╓
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_15/dropout/random_uniform/RandomUniformЛ
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_15/dropout/GreaterEqual/yы
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_15/dropout/GreaterEqualб
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_15/dropout/Castз
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_15/dropout/Mul_1┘
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╨
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul├
IdentityIdentitydropout_15/dropout/Mul_1:z:0%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_7_input
дu
■
__inference_call_798342

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_28_conv2d_readvariableop_resource: D
6sequential_7_conv2d_28_biasadd_readvariableop_resource: P
5sequential_7_conv2d_29_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_29_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_30_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_30_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_31_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_31_biasadd_readvariableop_resource:	АH
4sequential_7_dense_14_matmul_readvariableop_resource:
А@АD
5sequential_7_dense_14_biasadd_readvariableop_resource:	А:
'dense_15_matmul_readvariableop_resource:	А6
(dense_15_biasadd_readvariableop_resource:
identityИвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_28/BiasAdd/ReadVariableOpв,sequential_7/conv2d_28/Conv2D/ReadVariableOpв-sequential_7/conv2d_29/BiasAdd/ReadVariableOpв,sequential_7/conv2d_29/Conv2D/ReadVariableOpв-sequential_7/conv2d_30/BiasAdd/ReadVariableOpв,sequential_7/conv2d_30/Conv2D/ReadVariableOpв-sequential_7/conv2d_31/BiasAdd/ReadVariableOpв,sequential_7/conv2d_31/Conv2D/ReadVariableOpв,sequential_7/dense_14/BiasAdd/ReadVariableOpв+sequential_7/dense_14/MatMul/ReadVariableOpп
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2у
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1║
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_28/Conv2D/ReadVariableOpС
sequential_7/conv2d_28/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2
sequential_7/conv2d_28/Conv2D╤
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp▄
sequential_7/conv2d_28/BiasAddBiasAdd&sequential_7/conv2d_28/Conv2D:output:05sequential_7/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2 
sequential_7/conv2d_28/BiasAddЭ
sequential_7/conv2d_28/ReluRelu'sequential_7/conv2d_28/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_7/conv2d_28/Reluщ
%sequential_7/max_pooling2d_28/MaxPoolMaxPool)sequential_7/conv2d_28/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_28/MaxPool█
,sequential_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_29/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_29/Conv2DConv2D.sequential_7/max_pooling2d_28/MaxPool:output:04sequential_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2
sequential_7/conv2d_29/Conv2D╥
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_29/BiasAddBiasAdd&sequential_7/conv2d_29/Conv2D:output:05sequential_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2 
sequential_7/conv2d_29/BiasAddЮ
sequential_7/conv2d_29/ReluRelu'sequential_7/conv2d_29/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_7/conv2d_29/Reluъ
%sequential_7/max_pooling2d_29/MaxPoolMaxPool)sequential_7/conv2d_29/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_29/MaxPool▄
,sequential_7/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_30/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_30/Conv2DConv2D.sequential_7/max_pooling2d_29/MaxPool:output:04sequential_7/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_7/conv2d_30/Conv2D╥
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_30/BiasAddBiasAdd&sequential_7/conv2d_30/Conv2D:output:05sequential_7/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_7/conv2d_30/BiasAddЮ
sequential_7/conv2d_30/ReluRelu'sequential_7/conv2d_30/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_7/conv2d_30/Reluъ
%sequential_7/max_pooling2d_30/MaxPoolMaxPool)sequential_7/conv2d_30/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_30/MaxPool▄
,sequential_7/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_31/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_31/Conv2DConv2D.sequential_7/max_pooling2d_30/MaxPool:output:04sequential_7/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А*
paddingSAME*
strides
2
sequential_7/conv2d_31/Conv2D╥
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_31/BiasAddBiasAdd&sequential_7/conv2d_31/Conv2D:output:05sequential_7/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А2 
sequential_7/conv2d_31/BiasAddЮ
sequential_7/conv2d_31/ReluRelu'sequential_7/conv2d_31/BiasAdd:output:0*
T0*(
_output_shapes
:А		А2
sequential_7/conv2d_31/Reluъ
%sequential_7/max_pooling2d_31/MaxPoolMaxPool)sequential_7/conv2d_31/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_31/MaxPool│
 sequential_7/dropout_14/IdentityIdentity.sequential_7/max_pooling2d_31/MaxPool:output:0*
T0*(
_output_shapes
:АА2"
 sequential_7/dropout_14/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_7/flatten_7/Const╚
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_14/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0* 
_output_shapes
:
АА@2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_7/dense_14/MatMul/ReadVariableOp╧
sequential_7/dense_14/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_14/MatMul╧
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOp╥
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_14/BiasAddУ
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_14/Reluе
 sequential_7/dropout_15/IdentityIdentity(sequential_7/dense_14/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_7/dropout_15/Identityй
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_15/MatMul/ReadVariableOpй
dense_15/MatMulMatMul)sequential_7/dropout_15/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_15/MatMulз
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpЭ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_15/BiasAddt
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_15/Softmax°
IdentityIdentitydense_15/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_28/BiasAdd/ReadVariableOp-^sequential_7/conv2d_28/Conv2D/ReadVariableOp.^sequential_7/conv2d_29/BiasAdd/ReadVariableOp-^sequential_7/conv2d_29/Conv2D/ReadVariableOp.^sequential_7/conv2d_30/BiasAdd/ReadVariableOp-^sequential_7/conv2d_30/Conv2D/ReadVariableOp.^sequential_7/conv2d_31/BiasAdd/ReadVariableOp-^sequential_7/conv2d_31/Conv2D/ReadVariableOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp-sequential_7/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_28/Conv2D/ReadVariableOp,sequential_7/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp-sequential_7/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_29/Conv2D/ReadVariableOp,sequential_7/conv2d_29/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp-sequential_7/conv2d_30/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_30/Conv2D/ReadVariableOp,sequential_7/conv2d_30/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp-sequential_7/conv2d_31/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_31/Conv2D/ReadVariableOp,sequential_7/conv2d_31/Conv2D/ReadVariableOp2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
ї
d
+__inference_dropout_14_layer_call_fn_834381

inputs
identityИвStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_8322922
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╣
м
D__inference_dense_14_layer_call_and_return_conditional_losses_832176

inputs2
matmul_readvariableop_resource:
А@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relu╟
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
├
Ь
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834192

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
лн
·
@__inference_CNN3_layer_call_and_return_conditional_losses_833282

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_28_conv2d_readvariableop_resource: D
6sequential_7_conv2d_28_biasadd_readvariableop_resource: P
5sequential_7_conv2d_29_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_29_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_30_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_30_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_31_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_31_biasadd_readvariableop_resource:	АH
4sequential_7_dense_14_matmul_readvariableop_resource:
А@АD
5sequential_7_dense_14_biasadd_readvariableop_resource:	А:
'dense_15_matmul_readvariableop_resource:	А6
(dense_15_biasadd_readvariableop_resource:
identityИв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpв1sequential_7/batch_normalization_7/AssignNewValueв3sequential_7/batch_normalization_7/AssignNewValue_1вBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_28/BiasAdd/ReadVariableOpв,sequential_7/conv2d_28/Conv2D/ReadVariableOpв-sequential_7/conv2d_29/BiasAdd/ReadVariableOpв,sequential_7/conv2d_29/Conv2D/ReadVariableOpв-sequential_7/conv2d_30/BiasAdd/ReadVariableOpв,sequential_7/conv2d_30/Conv2D/ReadVariableOpв-sequential_7/conv2d_31/BiasAdd/ReadVariableOpв,sequential_7/conv2d_31/Conv2D/ReadVariableOpв,sequential_7/dense_14/BiasAdd/ReadVariableOpв+sequential_7/dense_14/MatMul/ReadVariableOpп
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2ы
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1╨
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<25
3sequential_7/batch_normalization_7/FusedBatchNormV3ё
1sequential_7/batch_normalization_7/AssignNewValueAssignVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource@sequential_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_7/batch_normalization_7/AssignNewValue¤
3sequential_7/batch_normalization_7/AssignNewValue_1AssignVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceDsequential_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0E^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_7/batch_normalization_7/AssignNewValue_1┌
,sequential_7/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_28/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_28/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_28/Conv2D╤
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpф
sequential_7/conv2d_28/BiasAddBiasAdd&sequential_7/conv2d_28/Conv2D:output:05sequential_7/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_28/BiasAddе
sequential_7/conv2d_28/ReluRelu'sequential_7/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_28/Reluё
%sequential_7/max_pooling2d_28/MaxPoolMaxPool)sequential_7/conv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_28/MaxPool█
,sequential_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_29/Conv2D/ReadVariableOpС
sequential_7/conv2d_29/Conv2DConv2D.sequential_7/max_pooling2d_28/MaxPool:output:04sequential_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_7/conv2d_29/Conv2D╥
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpх
sequential_7/conv2d_29/BiasAddBiasAdd&sequential_7/conv2d_29/Conv2D:output:05sequential_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_7/conv2d_29/BiasAddж
sequential_7/conv2d_29/ReluRelu'sequential_7/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_7/conv2d_29/ReluЄ
%sequential_7/max_pooling2d_29/MaxPoolMaxPool)sequential_7/conv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_29/MaxPool▄
,sequential_7/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_30/Conv2D/ReadVariableOpС
sequential_7/conv2d_30/Conv2DConv2D.sequential_7/max_pooling2d_29/MaxPool:output:04sequential_7/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_30/Conv2D╥
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpх
sequential_7/conv2d_30/BiasAddBiasAdd&sequential_7/conv2d_30/Conv2D:output:05sequential_7/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_30/BiasAddж
sequential_7/conv2d_30/ReluRelu'sequential_7/conv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_30/ReluЄ
%sequential_7/max_pooling2d_30/MaxPoolMaxPool)sequential_7/conv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_30/MaxPool▄
,sequential_7/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_31/Conv2D/ReadVariableOpС
sequential_7/conv2d_31/Conv2DConv2D.sequential_7/max_pooling2d_30/MaxPool:output:04sequential_7/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_7/conv2d_31/Conv2D╥
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpх
sequential_7/conv2d_31/BiasAddBiasAdd&sequential_7/conv2d_31/Conv2D:output:05sequential_7/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_7/conv2d_31/BiasAddж
sequential_7/conv2d_31/ReluRelu'sequential_7/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_7/conv2d_31/ReluЄ
%sequential_7/max_pooling2d_31/MaxPoolMaxPool)sequential_7/conv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_31/MaxPoolУ
%sequential_7/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2'
%sequential_7/dropout_14/dropout/Constь
#sequential_7/dropout_14/dropout/MulMul.sequential_7/max_pooling2d_31/MaxPool:output:0.sequential_7/dropout_14/dropout/Const:output:0*
T0*0
_output_shapes
:         А2%
#sequential_7/dropout_14/dropout/Mulм
%sequential_7/dropout_14/dropout/ShapeShape.sequential_7/max_pooling2d_31/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_14/dropout/ShapeЕ
<sequential_7/dropout_14/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_14/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_14/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_7/dropout_14/dropout/GreaterEqual/yз
,sequential_7/dropout_14/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_14/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_14/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2.
,sequential_7/dropout_14/dropout/GreaterEqual╨
$sequential_7/dropout_14/dropout/CastCast0sequential_7/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2&
$sequential_7/dropout_14/dropout/Castу
%sequential_7/dropout_14/dropout/Mul_1Mul'sequential_7/dropout_14/dropout/Mul:z:0(sequential_7/dropout_14/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2'
%sequential_7/dropout_14/dropout/Mul_1Н
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_14/dropout/Mul_1:z:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_7/dense_14/MatMul/ReadVariableOp╫
sequential_7/dense_14/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/MatMul╧
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOp┌
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/BiasAddЫ
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/ReluУ
%sequential_7/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_15/dropout/Const▐
#sequential_7/dropout_15/dropout/MulMul(sequential_7/dense_14/Relu:activations:0.sequential_7/dropout_15/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_7/dropout_15/dropout/Mulж
%sequential_7/dropout_15/dropout/ShapeShape(sequential_7/dense_14/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_15/dropout/Shape¤
<sequential_7/dropout_15/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_15/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_15/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_15/dropout/GreaterEqual/yЯ
,sequential_7/dropout_15/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_15/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_15/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_7/dropout_15/dropout/GreaterEqual╚
$sequential_7/dropout_15/dropout/CastCast0sequential_7/dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_7/dropout_15/dropout/Cast█
%sequential_7/dropout_15/dropout/Mul_1Mul'sequential_7/dropout_15/dropout/Mul:z:0(sequential_7/dropout_15/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_7/dropout_15/dropout/Mul_1й
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_15/MatMul/ReadVariableOp▒
dense_15/MatMulMatMul)sequential_7/dropout_15/dropout/Mul_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/MatMulз
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpе
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/BiasAdd|
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_15/Softmaxц
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul▌
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul╙
IdentityIdentitydense_15/Softmax:softmax:03^conv2d_28/kernel/Regularizer/Square/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^sequential_7/batch_normalization_7/AssignNewValue4^sequential_7/batch_normalization_7/AssignNewValue_1C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_28/BiasAdd/ReadVariableOp-^sequential_7/conv2d_28/Conv2D/ReadVariableOp.^sequential_7/conv2d_29/BiasAdd/ReadVariableOp-^sequential_7/conv2d_29/Conv2D/ReadVariableOp.^sequential_7/conv2d_30/BiasAdd/ReadVariableOp-^sequential_7/conv2d_30/Conv2D/ReadVariableOp.^sequential_7/conv2d_31/BiasAdd/ReadVariableOp-^sequential_7/conv2d_31/Conv2D/ReadVariableOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1sequential_7/batch_normalization_7/AssignNewValue1sequential_7/batch_normalization_7/AssignNewValue2j
3sequential_7/batch_normalization_7/AssignNewValue_13sequential_7/batch_normalization_7/AssignNewValue_12И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp-sequential_7/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_28/Conv2D/ReadVariableOp,sequential_7/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp-sequential_7/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_29/Conv2D/ReadVariableOp,sequential_7/conv2d_29/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp-sequential_7/conv2d_30/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_30/Conv2D/ReadVariableOp,sequential_7/conv2d_30/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp-sequential_7/conv2d_31/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_31/Conv2D/ReadVariableOp,sequential_7/conv2d_31/Conv2D/ReadVariableOp2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
б
Б
E__inference_conv2d_30_layer_call_and_return_conditional_losses_834325

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
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
▐
M
1__inference_max_pooling2d_29_layer_call_fn_831998

inputs
identityЄ
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
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_8319922
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
г
╢
%__inference_CNN3_layer_call_fn_833612
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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallп
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
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_CNN3_layer_call_and_return_conditional_losses_8328712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
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
┬
`
D__inference_lambda_7_layer_call_and_return_conditional_losses_832395

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
щ
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_834387

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ш
│
__inference_loss_fn_1_834473N
:dense_14_kernel_regularizer_square_readvariableop_resource:
А@А
identityИв1dense_14/kernel/Regularizer/Square/ReadVariableOpу
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_14_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulЪ
IdentityIdentity#dense_14/kernel/Regularizer/mul:z:02^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp
дu
■
__inference_call_798414

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_28_conv2d_readvariableop_resource: D
6sequential_7_conv2d_28_biasadd_readvariableop_resource: P
5sequential_7_conv2d_29_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_29_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_30_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_30_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_31_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_31_biasadd_readvariableop_resource:	АH
4sequential_7_dense_14_matmul_readvariableop_resource:
А@АD
5sequential_7_dense_14_biasadd_readvariableop_resource:	А:
'dense_15_matmul_readvariableop_resource:	А6
(dense_15_biasadd_readvariableop_resource:
identityИвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_28/BiasAdd/ReadVariableOpв,sequential_7/conv2d_28/Conv2D/ReadVariableOpв-sequential_7/conv2d_29/BiasAdd/ReadVariableOpв,sequential_7/conv2d_29/Conv2D/ReadVariableOpв-sequential_7/conv2d_30/BiasAdd/ReadVariableOpв,sequential_7/conv2d_30/Conv2D/ReadVariableOpв-sequential_7/conv2d_31/BiasAdd/ReadVariableOpв,sequential_7/conv2d_31/Conv2D/ReadVariableOpв,sequential_7/dense_14/BiasAdd/ReadVariableOpв+sequential_7/dense_14/MatMul/ReadVariableOpп
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2у
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:АKK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1║
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:АKK:::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_28/Conv2D/ReadVariableOpС
sequential_7/conv2d_28/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK *
paddingSAME*
strides
2
sequential_7/conv2d_28/Conv2D╤
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp▄
sequential_7/conv2d_28/BiasAddBiasAdd&sequential_7/conv2d_28/Conv2D:output:05sequential_7/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:АKK 2 
sequential_7/conv2d_28/BiasAddЭ
sequential_7/conv2d_28/ReluRelu'sequential_7/conv2d_28/BiasAdd:output:0*
T0*'
_output_shapes
:АKK 2
sequential_7/conv2d_28/Reluщ
%sequential_7/max_pooling2d_28/MaxPoolMaxPool)sequential_7/conv2d_28/Relu:activations:0*'
_output_shapes
:А%% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_28/MaxPool█
,sequential_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_29/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_29/Conv2DConv2D.sequential_7/max_pooling2d_28/MaxPool:output:04sequential_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А*
paddingSAME*
strides
2
sequential_7/conv2d_29/Conv2D╥
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_29/BiasAddBiasAdd&sequential_7/conv2d_29/Conv2D:output:05sequential_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А%%А2 
sequential_7/conv2d_29/BiasAddЮ
sequential_7/conv2d_29/ReluRelu'sequential_7/conv2d_29/BiasAdd:output:0*
T0*(
_output_shapes
:А%%А2
sequential_7/conv2d_29/Reluъ
%sequential_7/max_pooling2d_29/MaxPoolMaxPool)sequential_7/conv2d_29/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_29/MaxPool▄
,sequential_7/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_30/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_30/Conv2DConv2D.sequential_7/max_pooling2d_29/MaxPool:output:04sequential_7/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА*
paddingSAME*
strides
2
sequential_7/conv2d_30/Conv2D╥
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_30/BiasAddBiasAdd&sequential_7/conv2d_30/Conv2D:output:05sequential_7/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:АА2 
sequential_7/conv2d_30/BiasAddЮ
sequential_7/conv2d_30/ReluRelu'sequential_7/conv2d_30/BiasAdd:output:0*
T0*(
_output_shapes
:АА2
sequential_7/conv2d_30/Reluъ
%sequential_7/max_pooling2d_30/MaxPoolMaxPool)sequential_7/conv2d_30/Relu:activations:0*(
_output_shapes
:А		А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_30/MaxPool▄
,sequential_7/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_31/Conv2D/ReadVariableOpЙ
sequential_7/conv2d_31/Conv2DConv2D.sequential_7/max_pooling2d_30/MaxPool:output:04sequential_7/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А*
paddingSAME*
strides
2
sequential_7/conv2d_31/Conv2D╥
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp▌
sequential_7/conv2d_31/BiasAddBiasAdd&sequential_7/conv2d_31/Conv2D:output:05sequential_7/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:А		А2 
sequential_7/conv2d_31/BiasAddЮ
sequential_7/conv2d_31/ReluRelu'sequential_7/conv2d_31/BiasAdd:output:0*
T0*(
_output_shapes
:А		А2
sequential_7/conv2d_31/Reluъ
%sequential_7/max_pooling2d_31/MaxPoolMaxPool)sequential_7/conv2d_31/Relu:activations:0*(
_output_shapes
:АА*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_31/MaxPool│
 sequential_7/dropout_14/IdentityIdentity.sequential_7/max_pooling2d_31/MaxPool:output:0*
T0*(
_output_shapes
:АА2"
 sequential_7/dropout_14/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_7/flatten_7/Const╚
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_14/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0* 
_output_shapes
:
АА@2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_7/dense_14/MatMul/ReadVariableOp╧
sequential_7/dense_14/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_14/MatMul╧
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOp╥
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_14/BiasAddУ
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0* 
_output_shapes
:
АА2
sequential_7/dense_14/Reluе
 sequential_7/dropout_15/IdentityIdentity(sequential_7/dense_14/Relu:activations:0*
T0* 
_output_shapes
:
АА2"
 sequential_7/dropout_15/Identityй
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_15/MatMul/ReadVariableOpй
dense_15/MatMulMatMul)sequential_7/dropout_15/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_15/MatMulз
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpЭ
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	А2
dense_15/BiasAddt
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*
_output_shapes
:	А2
dense_15/Softmax°
IdentityIdentitydense_15/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_28/BiasAdd/ReadVariableOp-^sequential_7/conv2d_28/Conv2D/ReadVariableOp.^sequential_7/conv2d_29/BiasAdd/ReadVariableOp-^sequential_7/conv2d_29/Conv2D/ReadVariableOp.^sequential_7/conv2d_30/BiasAdd/ReadVariableOp-^sequential_7/conv2d_30/Conv2D/ReadVariableOp.^sequential_7/conv2d_31/BiasAdd/ReadVariableOp-^sequential_7/conv2d_31/Conv2D/ReadVariableOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp*
T0*
_output_shapes
:	А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:АKK: : : : : : : : : : : : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp-sequential_7/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_28/Conv2D/ReadVariableOp,sequential_7/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp-sequential_7/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_29/Conv2D/ReadVariableOp,sequential_7/conv2d_29/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp-sequential_7/conv2d_30/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_30/Conv2D/ReadVariableOp,sequential_7/conv2d_30/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp-sequential_7/conv2d_31/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_31/Conv2D/ReadVariableOp,sequential_7/conv2d_31/Conv2D/ReadVariableOp2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp:O K
'
_output_shapes
:АKK
 
_user_specified_nameinputs
╔
G
+__inference_dropout_15_layer_call_fn_834446

inputs
identity╩
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
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_8321872
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╤
в
*__inference_conv2d_30_layer_call_fn_834334

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallГ
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
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_30_layer_call_and_return_conditional_losses_8321192
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
иk
═
__inference__traced_save_834655
file_prefix.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop/
+savev2_conv2d_30_kernel_read_readvariableop-
)savev2_conv2d_30_bias_read_readvariableop/
+savev2_conv2d_31_kernel_read_readvariableop-
)savev2_conv2d_31_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop6
2savev2_adam_conv2d_28_kernel_m_read_readvariableop4
0savev2_adam_conv2d_28_bias_m_read_readvariableop6
2savev2_adam_conv2d_29_kernel_m_read_readvariableop4
0savev2_adam_conv2d_29_bias_m_read_readvariableop6
2savev2_adam_conv2d_30_kernel_m_read_readvariableop4
0savev2_adam_conv2d_30_bias_m_read_readvariableop6
2savev2_adam_conv2d_31_kernel_m_read_readvariableop4
0savev2_adam_conv2d_31_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop6
2savev2_adam_conv2d_28_kernel_v_read_readvariableop4
0savev2_adam_conv2d_28_bias_v_read_readvariableop6
2savev2_adam_conv2d_29_kernel_v_read_readvariableop4
0savev2_adam_conv2d_29_bias_v_read_readvariableop6
2savev2_adam_conv2d_30_kernel_v_read_readvariableop4
0savev2_adam_conv2d_30_bias_v_read_readvariableop6
2savev2_adam_conv2d_31_kernel_v_read_readvariableop4
0savev2_adam_conv2d_31_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╚
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*┌
value╨B═6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices№
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop+savev2_conv2d_30_kernel_read_readvariableop)savev2_conv2d_30_bias_read_readvariableop+savev2_conv2d_31_kernel_read_readvariableop)savev2_conv2d_31_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop2savev2_adam_conv2d_28_kernel_m_read_readvariableop0savev2_adam_conv2d_28_bias_m_read_readvariableop2savev2_adam_conv2d_29_kernel_m_read_readvariableop0savev2_adam_conv2d_29_bias_m_read_readvariableop2savev2_adam_conv2d_30_kernel_m_read_readvariableop0savev2_adam_conv2d_30_bias_m_read_readvariableop2savev2_adam_conv2d_31_kernel_m_read_readvariableop0savev2_adam_conv2d_31_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop2savev2_adam_conv2d_28_kernel_v_read_readvariableop0savev2_adam_conv2d_28_bias_v_read_readvariableop2savev2_adam_conv2d_29_kernel_v_read_readvariableop0savev2_adam_conv2d_29_bias_v_read_readvariableop2savev2_adam_conv2d_30_kernel_v_read_readvariableop0savev2_adam_conv2d_30_bias_v_read_readvariableop2savev2_adam_conv2d_31_kernel_v_read_readvariableop0savev2_adam_conv2d_31_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0* 
_input_shapesэ
ъ: :	А:: : : : : ::: : : А:А:АА:А:АА:А:
А@А:А::: : : : :	А:::: : : А:А:АА:А:АА:А:
А@А:А:	А:::: : : А:А:АА:А:АА:А:
А@А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
А@А:!

_output_shapes	
:А: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	А: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :- )
'
_output_shapes
: А:!!

_output_shapes	
:А:."*
(
_output_shapes
:АА:!#

_output_shapes	
:А:.$*
(
_output_shapes
:АА:!%

_output_shapes	
:А:&&"
 
_output_shapes
:
А@А:!'

_output_shapes	
:А:%(!

_output_shapes
:	А: )

_output_shapes
:: *

_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: :-.)
'
_output_shapes
: А:!/

_output_shapes	
:А:.0*
(
_output_shapes
:АА:!1

_output_shapes	
:А:.2*
(
_output_shapes
:АА:!3

_output_shapes	
:А:&4"
 
_output_shapes
:
А@А:!5

_output_shapes	
:А:6

_output_shapes
: 
Э
А
E__inference_conv2d_29_layer_call_and_return_conditional_losses_832101

inputs9
conv2d_readvariableop_resource: А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%А2

Identity"
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
б
Б
E__inference_conv2d_31_layer_call_and_return_conditional_losses_834345

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╒
d
+__inference_dropout_15_layer_call_fn_834451

inputs
identityИвStatefulPartitionedCallт
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
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_8322532
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ъ
╗
__inference_loss_fn_0_834462U
;conv2d_28_kernel_regularizer_square_readvariableop_resource: 
identityИв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpь
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_28_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mulЬ
IdentityIdentity$conv2d_28/kernel/Regularizer/mul:z:03^conv2d_28/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp
╟т
ч!
"__inference__traced_restore_834824
file_prefix3
 assignvariableop_dense_15_kernel:	А.
 assignvariableop_1_dense_15_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_7_gamma:;
-assignvariableop_8_batch_normalization_7_beta:=
#assignvariableop_9_conv2d_28_kernel: 0
"assignvariableop_10_conv2d_28_bias: ?
$assignvariableop_11_conv2d_29_kernel: А1
"assignvariableop_12_conv2d_29_bias:	А@
$assignvariableop_13_conv2d_30_kernel:АА1
"assignvariableop_14_conv2d_30_bias:	А@
$assignvariableop_15_conv2d_31_kernel:АА1
"assignvariableop_16_conv2d_31_bias:	А7
#assignvariableop_17_dense_14_kernel:
А@А0
!assignvariableop_18_dense_14_bias:	АC
5assignvariableop_19_batch_normalization_7_moving_mean:G
9assignvariableop_20_batch_normalization_7_moving_variance:#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_15_kernel_m:	А6
(assignvariableop_26_adam_dense_15_bias_m:D
6assignvariableop_27_adam_batch_normalization_7_gamma_m:C
5assignvariableop_28_adam_batch_normalization_7_beta_m:E
+assignvariableop_29_adam_conv2d_28_kernel_m: 7
)assignvariableop_30_adam_conv2d_28_bias_m: F
+assignvariableop_31_adam_conv2d_29_kernel_m: А8
)assignvariableop_32_adam_conv2d_29_bias_m:	АG
+assignvariableop_33_adam_conv2d_30_kernel_m:АА8
)assignvariableop_34_adam_conv2d_30_bias_m:	АG
+assignvariableop_35_adam_conv2d_31_kernel_m:АА8
)assignvariableop_36_adam_conv2d_31_bias_m:	А>
*assignvariableop_37_adam_dense_14_kernel_m:
А@А7
(assignvariableop_38_adam_dense_14_bias_m:	А=
*assignvariableop_39_adam_dense_15_kernel_v:	А6
(assignvariableop_40_adam_dense_15_bias_v:D
6assignvariableop_41_adam_batch_normalization_7_gamma_v:C
5assignvariableop_42_adam_batch_normalization_7_beta_v:E
+assignvariableop_43_adam_conv2d_28_kernel_v: 7
)assignvariableop_44_adam_conv2d_28_bias_v: F
+assignvariableop_45_adam_conv2d_29_kernel_v: А8
)assignvariableop_46_adam_conv2d_29_bias_v:	АG
+assignvariableop_47_adam_conv2d_30_kernel_v:АА8
)assignvariableop_48_adam_conv2d_30_bias_v:	АG
+assignvariableop_49_adam_conv2d_31_kernel_v:АА8
)assignvariableop_50_adam_conv2d_31_bias_v:	А>
*assignvariableop_51_adam_dense_14_kernel_v:
А@А7
(assignvariableop_52_adam_dense_14_bias_v:	А
identity_54ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╬
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*┌
value╨B═6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names·
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╝
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ю
_output_shapes█
╪::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2б
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4г
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5в
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6к
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7│
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_7_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8▓
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_7_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9и
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_28_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10к
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_28_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11м
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_29_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12к
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_29_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13м
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_30_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14к
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_30_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15м
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_31_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16к
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_31_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17л
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_14_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18й
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_14_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╜
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_7_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20┴
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_7_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21б
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22б
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23г
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24г
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▓
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_15_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_15_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╛
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_7_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╜
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_7_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29│
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_28_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▒
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_28_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31│
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_29_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_29_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_30_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_30_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35│
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_31_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▒
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_31_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_14_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38░
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_14_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_15_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_15_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╛
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_7_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╜
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_7_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43│
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_28_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44▒
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_28_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45│
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_29_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_29_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_30_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_30_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49│
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_31_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50▒
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_31_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51▓
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_14_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52░
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_14_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpь	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_53▀	
Identity_54IdentityIdentity_53:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_54"#
identity_54Identity_54:output:0*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
┐
│
E__inference_conv2d_28_layer_call_and_return_conditional_losses_832083

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu╧
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┬
`
D__inference_lambda_7_layer_call_and_return_conditional_losses_832037

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ё
М
-__inference_sequential_7_layer_call_fn_834092
lambda_7_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCalllambda_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_8324912
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_7_input
он
√
@__inference_CNN3_layer_call_and_return_conditional_losses_833464
input_1H
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_28_conv2d_readvariableop_resource: D
6sequential_7_conv2d_28_biasadd_readvariableop_resource: P
5sequential_7_conv2d_29_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_29_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_30_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_30_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_31_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_31_biasadd_readvariableop_resource:	АH
4sequential_7_dense_14_matmul_readvariableop_resource:
А@АD
5sequential_7_dense_14_biasadd_readvariableop_resource:	А:
'dense_15_matmul_readvariableop_resource:	А6
(dense_15_biasadd_readvariableop_resource:
identityИв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpв1sequential_7/batch_normalization_7/AssignNewValueв3sequential_7/batch_normalization_7/AssignNewValue_1вBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_28/BiasAdd/ReadVariableOpв,sequential_7/conv2d_28/Conv2D/ReadVariableOpв-sequential_7/conv2d_29/BiasAdd/ReadVariableOpв,sequential_7/conv2d_29/Conv2D/ReadVariableOpв-sequential_7/conv2d_30/BiasAdd/ReadVariableOpв,sequential_7/conv2d_30/Conv2D/ReadVariableOpв-sequential_7/conv2d_31/BiasAdd/ReadVariableOpв,sequential_7/conv2d_31/Conv2D/ReadVariableOpв,sequential_7/dense_14/BiasAdd/ReadVariableOpв+sequential_7/dense_14/MatMul/ReadVariableOpп
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2ь
#sequential_7/lambda_7/strided_sliceStridedSliceinput_12sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1╨
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<25
3sequential_7/batch_normalization_7/FusedBatchNormV3ё
1sequential_7/batch_normalization_7/AssignNewValueAssignVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource@sequential_7/batch_normalization_7/FusedBatchNormV3:batch_mean:0C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_7/batch_normalization_7/AssignNewValue¤
3sequential_7/batch_normalization_7/AssignNewValue_1AssignVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceDsequential_7/batch_normalization_7/FusedBatchNormV3:batch_variance:0E^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_7/batch_normalization_7/AssignNewValue_1┌
,sequential_7/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_28/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_28/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_28/Conv2D╤
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpф
sequential_7/conv2d_28/BiasAddBiasAdd&sequential_7/conv2d_28/Conv2D:output:05sequential_7/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_28/BiasAddе
sequential_7/conv2d_28/ReluRelu'sequential_7/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_28/Reluё
%sequential_7/max_pooling2d_28/MaxPoolMaxPool)sequential_7/conv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_28/MaxPool█
,sequential_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_29/Conv2D/ReadVariableOpС
sequential_7/conv2d_29/Conv2DConv2D.sequential_7/max_pooling2d_28/MaxPool:output:04sequential_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_7/conv2d_29/Conv2D╥
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpх
sequential_7/conv2d_29/BiasAddBiasAdd&sequential_7/conv2d_29/Conv2D:output:05sequential_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_7/conv2d_29/BiasAddж
sequential_7/conv2d_29/ReluRelu'sequential_7/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_7/conv2d_29/ReluЄ
%sequential_7/max_pooling2d_29/MaxPoolMaxPool)sequential_7/conv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_29/MaxPool▄
,sequential_7/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_30/Conv2D/ReadVariableOpС
sequential_7/conv2d_30/Conv2DConv2D.sequential_7/max_pooling2d_29/MaxPool:output:04sequential_7/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_30/Conv2D╥
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpх
sequential_7/conv2d_30/BiasAddBiasAdd&sequential_7/conv2d_30/Conv2D:output:05sequential_7/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_30/BiasAddж
sequential_7/conv2d_30/ReluRelu'sequential_7/conv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_30/ReluЄ
%sequential_7/max_pooling2d_30/MaxPoolMaxPool)sequential_7/conv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_30/MaxPool▄
,sequential_7/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_31/Conv2D/ReadVariableOpС
sequential_7/conv2d_31/Conv2DConv2D.sequential_7/max_pooling2d_30/MaxPool:output:04sequential_7/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_7/conv2d_31/Conv2D╥
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpх
sequential_7/conv2d_31/BiasAddBiasAdd&sequential_7/conv2d_31/Conv2D:output:05sequential_7/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_7/conv2d_31/BiasAddж
sequential_7/conv2d_31/ReluRelu'sequential_7/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_7/conv2d_31/ReluЄ
%sequential_7/max_pooling2d_31/MaxPoolMaxPool)sequential_7/conv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_31/MaxPoolУ
%sequential_7/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2'
%sequential_7/dropout_14/dropout/Constь
#sequential_7/dropout_14/dropout/MulMul.sequential_7/max_pooling2d_31/MaxPool:output:0.sequential_7/dropout_14/dropout/Const:output:0*
T0*0
_output_shapes
:         А2%
#sequential_7/dropout_14/dropout/Mulм
%sequential_7/dropout_14/dropout/ShapeShape.sequential_7/max_pooling2d_31/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_14/dropout/ShapeЕ
<sequential_7/dropout_14/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_14/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_14/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_7/dropout_14/dropout/GreaterEqual/yз
,sequential_7/dropout_14/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_14/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_14/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2.
,sequential_7/dropout_14/dropout/GreaterEqual╨
$sequential_7/dropout_14/dropout/CastCast0sequential_7/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2&
$sequential_7/dropout_14/dropout/Castу
%sequential_7/dropout_14/dropout/Mul_1Mul'sequential_7/dropout_14/dropout/Mul:z:0(sequential_7/dropout_14/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2'
%sequential_7/dropout_14/dropout/Mul_1Н
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_14/dropout/Mul_1:z:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_7/dense_14/MatMul/ReadVariableOp╫
sequential_7/dense_14/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/MatMul╧
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOp┌
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/BiasAddЫ
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/ReluУ
%sequential_7/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_7/dropout_15/dropout/Const▐
#sequential_7/dropout_15/dropout/MulMul(sequential_7/dense_14/Relu:activations:0.sequential_7/dropout_15/dropout/Const:output:0*
T0*(
_output_shapes
:         А2%
#sequential_7/dropout_15/dropout/Mulж
%sequential_7/dropout_15/dropout/ShapeShape(sequential_7/dense_14/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dropout_15/dropout/Shape¤
<sequential_7/dropout_15/dropout/random_uniform/RandomUniformRandomUniform.sequential_7/dropout_15/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02>
<sequential_7/dropout_15/dropout/random_uniform/RandomUniformе
.sequential_7/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_7/dropout_15/dropout/GreaterEqual/yЯ
,sequential_7/dropout_15/dropout/GreaterEqualGreaterEqualEsequential_7/dropout_15/dropout/random_uniform/RandomUniform:output:07sequential_7/dropout_15/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2.
,sequential_7/dropout_15/dropout/GreaterEqual╚
$sequential_7/dropout_15/dropout/CastCast0sequential_7/dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2&
$sequential_7/dropout_15/dropout/Cast█
%sequential_7/dropout_15/dropout/Mul_1Mul'sequential_7/dropout_15/dropout/Mul:z:0(sequential_7/dropout_15/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2'
%sequential_7/dropout_15/dropout/Mul_1й
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_15/MatMul/ReadVariableOp▒
dense_15/MatMulMatMul)sequential_7/dropout_15/dropout/Mul_1:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/MatMulз
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpе
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/BiasAdd|
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_15/Softmaxц
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul▌
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul╙
IdentityIdentitydense_15/Softmax:softmax:03^conv2d_28/kernel/Regularizer/Square/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp2^sequential_7/batch_normalization_7/AssignNewValue4^sequential_7/batch_normalization_7/AssignNewValue_1C^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_28/BiasAdd/ReadVariableOp-^sequential_7/conv2d_28/Conv2D/ReadVariableOp.^sequential_7/conv2d_29/BiasAdd/ReadVariableOp-^sequential_7/conv2d_29/Conv2D/ReadVariableOp.^sequential_7/conv2d_30/BiasAdd/ReadVariableOp-^sequential_7/conv2d_30/Conv2D/ReadVariableOp.^sequential_7/conv2d_31/BiasAdd/ReadVariableOp-^sequential_7/conv2d_31/Conv2D/ReadVariableOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2f
1sequential_7/batch_normalization_7/AssignNewValue1sequential_7/batch_normalization_7/AssignNewValue2j
3sequential_7/batch_normalization_7/AssignNewValue_13sequential_7/batch_normalization_7/AssignNewValue_12И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp-sequential_7/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_28/Conv2D/ReadVariableOp,sequential_7/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp-sequential_7/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_29/Conv2D/ReadVariableOp,sequential_7/conv2d_29/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp-sequential_7/conv2d_30/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_30/Conv2D/ReadVariableOp,sequential_7/conv2d_30/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp-sequential_7/conv2d_31/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_31/Conv2D/ReadVariableOp,sequential_7/conv2d_31/Conv2D/ReadVariableOp2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
╫
F
*__inference_flatten_7_layer_call_fn_834392

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_8321572
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
╬
б
*__inference_conv2d_29_layer_call_fn_834314

inputs"
unknown: А
	unknown_0:	А
identityИвStatefulPartitionedCallГ
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
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_8321012
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         %%А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         %% : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         %% 
 
_user_specified_nameinputs
є
М
-__inference_sequential_7_layer_call_fn_833993
lambda_7_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: А
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCalllambda_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_8322022
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_7_input
┬
`
D__inference_lambda_7_layer_call_and_return_conditional_losses_834128

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
е
╢
%__inference_CNN3_layer_call_fn_833501
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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCall▒
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
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_CNN3_layer_call_and_return_conditional_losses_8327372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
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
нm
ї
H__inference_sequential_7_layer_call_and_return_conditional_losses_833701

inputs;
-batch_normalization_7_readvariableop_resource:=
/batch_normalization_7_readvariableop_1_resource:L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_28_conv2d_readvariableop_resource: 7
)conv2d_28_biasadd_readvariableop_resource: C
(conv2d_29_conv2d_readvariableop_resource: А8
)conv2d_29_biasadd_readvariableop_resource:	АD
(conv2d_30_conv2d_readvariableop_resource:АА8
)conv2d_30_biasadd_readvariableop_resource:	АD
(conv2d_31_conv2d_readvariableop_resource:АА8
)conv2d_31_biasadd_readvariableop_resource:	А;
'dense_14_matmul_readvariableop_resource:
А@А7
(dense_14_biasadd_readvariableop_resource:	А
identityИв5batch_normalization_7/FusedBatchNormV3/ReadVariableOpв7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_7/ReadVariableOpв&batch_normalization_7/ReadVariableOp_1в conv2d_28/BiasAdd/ReadVariableOpвconv2d_28/Conv2D/ReadVariableOpв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв conv2d_29/BiasAdd/ReadVariableOpвconv2d_29/Conv2D/ReadVariableOpв conv2d_30/BiasAdd/ReadVariableOpвconv2d_30/Conv2D/ReadVariableOpв conv2d_31/BiasAdd/ReadVariableOpвconv2d_31/Conv2D/ReadVariableOpвdense_14/BiasAdd/ReadVariableOpвdense_14/MatMul/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpХ
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_7/strided_slice/stackЩ
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_7/strided_slice/stack_1Щ
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_7/strided_slice/stack_2к
lambda_7/strided_sliceStridedSliceinputs%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_7/strided_slice╢
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_7/ReadVariableOp╝
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1щ
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ч
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3lambda_7/strided_slice:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3│
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_28/Conv2D/ReadVariableOpх
conv2d_28/Conv2DConv2D*batch_normalization_7/FusedBatchNormV3:y:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_28/Conv2Dк
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp░
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_28/BiasAdd~
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_28/Relu╩
max_pooling2d_28/MaxPoolMaxPoolconv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_28/MaxPool┤
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02!
conv2d_29/Conv2D/ReadVariableOp▌
conv2d_29/Conv2DConv2D!max_pooling2d_28/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
conv2d_29/Conv2Dл
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp▒
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2
conv2d_29/BiasAdd
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
conv2d_29/Relu╦
max_pooling2d_29/MaxPoolMaxPoolconv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_29/MaxPool╡
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_30/Conv2D/ReadVariableOp▌
conv2d_30/Conv2DConv2D!max_pooling2d_29/MaxPool:output:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_30/Conv2Dл
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp▒
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_30/BiasAdd
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_30/Relu╦
max_pooling2d_30/MaxPoolMaxPoolconv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_30/MaxPool╡
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02!
conv2d_31/Conv2D/ReadVariableOp▌
conv2d_31/Conv2DConv2D!max_pooling2d_30/MaxPool:output:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
conv2d_31/Conv2Dл
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp▒
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
conv2d_31/BiasAdd
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
conv2d_31/Relu╦
max_pooling2d_31/MaxPoolMaxPoolconv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_31/MaxPoolФ
dropout_14/IdentityIdentity!max_pooling2d_31/MaxPool:output:0*
T0*0
_output_shapes
:         А2
dropout_14/Identitys
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_7/ConstЬ
flatten_7/ReshapeReshapedropout_14/Identity:output:0flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_7/Reshapeк
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_14/MatMul/ReadVariableOpг
dense_14/MatMulMatMulflatten_7/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_14/MatMulи
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_14/BiasAdd/ReadVariableOpж
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_14/BiasAddt
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_14/ReluЖ
dropout_15/IdentityIdentitydense_14/Relu:activations:0*
T0*(
_output_shapes
:         А2
dropout_15/Identity┘
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╨
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulє
IdentityIdentitydropout_15/Identity:output:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
╤
в
*__inference_conv2d_31_layer_call_fn_834354

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_31_layer_call_and_return_conditional_losses_8321372
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         		А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
╓U
к
H__inference_sequential_7_layer_call_and_return_conditional_losses_832491

inputs*
batch_normalization_7_832437:*
batch_normalization_7_832439:*
batch_normalization_7_832441:*
batch_normalization_7_832443:*
conv2d_28_832446: 
conv2d_28_832448: +
conv2d_29_832452: А
conv2d_29_832454:	А,
conv2d_30_832458:АА
conv2d_30_832460:	А,
conv2d_31_832464:АА
conv2d_31_832466:	А#
dense_14_832472:
А@А
dense_14_832474:	А
identityИв-batch_normalization_7/StatefulPartitionedCallв!conv2d_28/StatefulPartitionedCallв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв!conv2d_29/StatefulPartitionedCallв!conv2d_30/StatefulPartitionedCallв!conv2d_31/StatefulPartitionedCallв dense_14/StatefulPartitionedCallв1dense_14/kernel/Regularizer/Square/ReadVariableOpв"dropout_14/StatefulPartitionedCallв"dropout_15/StatefulPartitionedCallс
lambda_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_lambda_7_layer_call_and_return_conditional_losses_8323952
lambda_7/PartitionedCall╗
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0batch_normalization_7_832437batch_normalization_7_832439batch_normalization_7_832441batch_normalization_7_832443*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8323682/
-batch_normalization_7/StatefulPartitionedCall╓
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0conv2d_28_832446conv2d_28_832448*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_28_layer_call_and_return_conditional_losses_8320832#
!conv2d_28/StatefulPartitionedCallЭ
 max_pooling2d_28/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         %% * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_8319802"
 max_pooling2d_28/PartitionedCall╩
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_28/PartitionedCall:output:0conv2d_29_832452conv2d_29_832454*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_29_layer_call_and_return_conditional_losses_8321012#
!conv2d_29/StatefulPartitionedCallЮ
 max_pooling2d_29/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_8319922"
 max_pooling2d_29/PartitionedCall╩
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_29/PartitionedCall:output:0conv2d_30_832458conv2d_30_832460*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_30_layer_call_and_return_conditional_losses_8321192#
!conv2d_30/StatefulPartitionedCallЮ
 max_pooling2d_30/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_8320042"
 max_pooling2d_30/PartitionedCall╩
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_30/PartitionedCall:output:0conv2d_31_832464conv2d_31_832466*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_conv2d_31_layer_call_and_return_conditional_losses_8321372#
!conv2d_31/StatefulPartitionedCallЮ
 max_pooling2d_31/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_8320162"
 max_pooling2d_31/PartitionedCallг
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_8322922$
"dropout_14/StatefulPartitionedCallВ
flatten_7/PartitionedCallPartitionedCall+dropout_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_flatten_7_layer_call_and_return_conditional_losses_8321572
flatten_7/PartitionedCall╢
 dense_14/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_14_832472dense_14_832474*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_8321762"
 dense_14/StatefulPartitionedCall└
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_15_layer_call_and_return_conditional_losses_8322532$
"dropout_15/StatefulPartitionedCall┴
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_28_832446*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╕
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_14_832472* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mulЦ
IdentityIdentity+dropout_15/StatefulPartitionedCall:output:0.^batch_normalization_7/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall2^dense_14/kernel/Regularizer/Square/ReadVariableOp#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
м
h
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_832016

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Е
╡
$__inference_signature_wrapper_833100
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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallТ
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
*2
config_proto" 

CPU

GPU2 *0J 8В **
f%R#
!__inference__wrapped_model_8318482
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
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
с
E
)__inference_lambda_7_layer_call_fn_834138

inputs
identity╧
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
 *2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_lambda_7_layer_call_and_return_conditional_losses_8323952
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
а
╡
%__inference_CNN3_layer_call_fn_833575

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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallо
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
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_CNN3_layer_call_and_return_conditional_losses_8328712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Фw
■
__inference_call_796471

inputsH
:sequential_7_batch_normalization_7_readvariableop_resource:J
<sequential_7_batch_normalization_7_readvariableop_1_resource:Y
Ksequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:[
Msequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_7_conv2d_28_conv2d_readvariableop_resource: D
6sequential_7_conv2d_28_biasadd_readvariableop_resource: P
5sequential_7_conv2d_29_conv2d_readvariableop_resource: АE
6sequential_7_conv2d_29_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_30_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_30_biasadd_readvariableop_resource:	АQ
5sequential_7_conv2d_31_conv2d_readvariableop_resource:ААE
6sequential_7_conv2d_31_biasadd_readvariableop_resource:	АH
4sequential_7_dense_14_matmul_readvariableop_resource:
А@АD
5sequential_7_dense_14_biasadd_readvariableop_resource:	А:
'dense_15_matmul_readvariableop_resource:	А6
(dense_15_biasadd_readvariableop_resource:
identityИвdense_15/BiasAdd/ReadVariableOpвdense_15/MatMul/ReadVariableOpвBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpвDsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1в1sequential_7/batch_normalization_7/ReadVariableOpв3sequential_7/batch_normalization_7/ReadVariableOp_1в-sequential_7/conv2d_28/BiasAdd/ReadVariableOpв,sequential_7/conv2d_28/Conv2D/ReadVariableOpв-sequential_7/conv2d_29/BiasAdd/ReadVariableOpв,sequential_7/conv2d_29/Conv2D/ReadVariableOpв-sequential_7/conv2d_30/BiasAdd/ReadVariableOpв,sequential_7/conv2d_30/Conv2D/ReadVariableOpв-sequential_7/conv2d_31/BiasAdd/ReadVariableOpв,sequential_7/conv2d_31/Conv2D/ReadVariableOpв,sequential_7/dense_14/BiasAdd/ReadVariableOpв+sequential_7/dense_14/MatMul/ReadVariableOpп
)sequential_7/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_7/lambda_7/strided_slice/stack│
+sequential_7/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_7/lambda_7/strided_slice/stack_1│
+sequential_7/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_7/lambda_7/strided_slice/stack_2ы
#sequential_7/lambda_7/strided_sliceStridedSliceinputs2sequential_7/lambda_7/strided_slice/stack:output:04sequential_7/lambda_7/strided_slice/stack_1:output:04sequential_7/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_7/lambda_7/strided_slice▌
1sequential_7/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_7_batch_normalization_7_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_7/batch_normalization_7/ReadVariableOpу
3sequential_7/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_7_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_7/batch_normalization_7/ReadVariableOp_1Р
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_7_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_7/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3,sequential_7/lambda_7/strided_slice:output:09sequential_7/batch_normalization_7/ReadVariableOp:value:0;sequential_7/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oГ:*
is_training( 25
3sequential_7/batch_normalization_7/FusedBatchNormV3┌
,sequential_7/conv2d_28/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_28/Conv2D/ReadVariableOpЩ
sequential_7/conv2d_28/Conv2DConv2D7sequential_7/batch_normalization_7/FusedBatchNormV3:y:04sequential_7/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_7/conv2d_28/Conv2D╤
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_28/BiasAdd/ReadVariableOpф
sequential_7/conv2d_28/BiasAddBiasAdd&sequential_7/conv2d_28/Conv2D:output:05sequential_7/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_7/conv2d_28/BiasAddе
sequential_7/conv2d_28/ReluRelu'sequential_7/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_7/conv2d_28/Reluё
%sequential_7/max_pooling2d_28/MaxPoolMaxPool)sequential_7/conv2d_28/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_28/MaxPool█
,sequential_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
: А*
dtype02.
,sequential_7/conv2d_29/Conv2D/ReadVariableOpС
sequential_7/conv2d_29/Conv2DConv2D.sequential_7/max_pooling2d_28/MaxPool:output:04sequential_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
2
sequential_7/conv2d_29/Conv2D╥
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_29/BiasAdd/ReadVariableOpх
sequential_7/conv2d_29/BiasAddBiasAdd&sequential_7/conv2d_29/Conv2D:output:05sequential_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А2 
sequential_7/conv2d_29/BiasAddж
sequential_7/conv2d_29/ReluRelu'sequential_7/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А2
sequential_7/conv2d_29/ReluЄ
%sequential_7/max_pooling2d_29/MaxPoolMaxPool)sequential_7/conv2d_29/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_29/MaxPool▄
,sequential_7/conv2d_30/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_30/Conv2D/ReadVariableOpС
sequential_7/conv2d_30/Conv2DConv2D.sequential_7/max_pooling2d_29/MaxPool:output:04sequential_7/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
sequential_7/conv2d_30/Conv2D╥
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_30/BiasAdd/ReadVariableOpх
sequential_7/conv2d_30/BiasAddBiasAdd&sequential_7/conv2d_30/Conv2D:output:05sequential_7/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2 
sequential_7/conv2d_30/BiasAddж
sequential_7/conv2d_30/ReluRelu'sequential_7/conv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
sequential_7/conv2d_30/ReluЄ
%sequential_7/max_pooling2d_30/MaxPoolMaxPool)sequential_7/conv2d_30/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_30/MaxPool▄
,sequential_7/conv2d_31/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_31_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02.
,sequential_7/conv2d_31/Conv2D/ReadVariableOpС
sequential_7/conv2d_31/Conv2DConv2D.sequential_7/max_pooling2d_30/MaxPool:output:04sequential_7/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
2
sequential_7/conv2d_31/Conv2D╥
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_31_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-sequential_7/conv2d_31/BiasAdd/ReadVariableOpх
sequential_7/conv2d_31/BiasAddBiasAdd&sequential_7/conv2d_31/Conv2D:output:05sequential_7/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2 
sequential_7/conv2d_31/BiasAddж
sequential_7/conv2d_31/ReluRelu'sequential_7/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
sequential_7/conv2d_31/ReluЄ
%sequential_7/max_pooling2d_31/MaxPoolMaxPool)sequential_7/conv2d_31/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_31/MaxPool╗
 sequential_7/dropout_14/IdentityIdentity.sequential_7/max_pooling2d_31/MaxPool:output:0*
T0*0
_output_shapes
:         А2"
 sequential_7/dropout_14/IdentityН
sequential_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
sequential_7/flatten_7/Const╨
sequential_7/flatten_7/ReshapeReshape)sequential_7/dropout_14/Identity:output:0%sequential_7/flatten_7/Const:output:0*
T0*(
_output_shapes
:         А@2 
sequential_7/flatten_7/Reshape╤
+sequential_7/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_7_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02-
+sequential_7/dense_14/MatMul/ReadVariableOp╫
sequential_7/dense_14/MatMulMatMul'sequential_7/flatten_7/Reshape:output:03sequential_7/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/MatMul╧
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOp┌
sequential_7/dense_14/BiasAddBiasAdd&sequential_7/dense_14/MatMul:product:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/BiasAddЫ
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
sequential_7/dense_14/Reluн
 sequential_7/dropout_15/IdentityIdentity(sequential_7/dense_14/Relu:activations:0*
T0*(
_output_shapes
:         А2"
 sequential_7/dropout_15/Identityй
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_15/MatMul/ReadVariableOp▒
dense_15/MatMulMatMul)sequential_7/dropout_15/Identity:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/MatMulз
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOpе
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_15/BiasAdd|
dense_15/SoftmaxSoftmaxdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_15/SoftmaxА
IdentityIdentitydense_15/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOpC^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_7/batch_normalization_7/ReadVariableOp4^sequential_7/batch_normalization_7/ReadVariableOp_1.^sequential_7/conv2d_28/BiasAdd/ReadVariableOp-^sequential_7/conv2d_28/Conv2D/ReadVariableOp.^sequential_7/conv2d_29/BiasAdd/ReadVariableOp-^sequential_7/conv2d_29/Conv2D/ReadVariableOp.^sequential_7/conv2d_30/BiasAdd/ReadVariableOp-^sequential_7/conv2d_30/Conv2D/ReadVariableOp.^sequential_7/conv2d_31/BiasAdd/ReadVariableOp-^sequential_7/conv2d_31/Conv2D/ReadVariableOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp,^sequential_7/dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2И
Bsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2М
Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_7/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12f
1sequential_7/batch_normalization_7/ReadVariableOp1sequential_7/batch_normalization_7/ReadVariableOp2j
3sequential_7/batch_normalization_7/ReadVariableOp_13sequential_7/batch_normalization_7/ReadVariableOp_12^
-sequential_7/conv2d_28/BiasAdd/ReadVariableOp-sequential_7/conv2d_28/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_28/Conv2D/ReadVariableOp,sequential_7/conv2d_28/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_29/BiasAdd/ReadVariableOp-sequential_7/conv2d_29/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_29/Conv2D/ReadVariableOp,sequential_7/conv2d_29/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_30/BiasAdd/ReadVariableOp-sequential_7/conv2d_30/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_30/Conv2D/ReadVariableOp,sequential_7/conv2d_30/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_31/BiasAdd/ReadVariableOp-sequential_7/conv2d_31/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_31/Conv2D/ReadVariableOp,sequential_7/conv2d_31/Conv2D/ReadVariableOp2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_7/dense_14/MatMul/ReadVariableOp+sequential_7/dense_14/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
┬
`
D__inference_lambda_7_layer_call_and_return_conditional_losses_834120

inputs
identityГ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЗ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1З
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2¤
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
strided_slicer
IdentityIdentitystrided_slice:output:0*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         KK:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
щ
a
E__inference_flatten_7_layer_call_and_return_conditional_losses_832157

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ў
d
F__inference_dropout_15_layer_call_and_return_conditional_losses_832187

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
з
Щ
)__inference_dense_14_layer_call_fn_834424

inputs
unknown:
А@А
	unknown_0:	А
identityИвStatefulPartitionedCall·
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
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_8321762
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
в
╡
%__inference_CNN3_layer_call_fn_833538

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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCall░
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
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_CNN3_layer_call_and_return_conditional_losses_8327372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ў
e
F__inference_dropout_14_layer_call_and_return_conditional_losses_832292

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╟
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
д
╤
6__inference_batch_normalization_7_layer_call_fn_834262

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_8323682
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         KK2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         KK: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
щ
G
+__inference_dropout_14_layer_call_fn_834376

inputs
identity╥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_8321492
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
█
Д
-__inference_sequential_7_layer_call_fn_834026

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
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А

unknown_11:
А@А

unknown_12:	А
identityИвStatefulPartitionedCallЭ
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
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_8322022
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Г(
■
@__inference_CNN3_layer_call_and_return_conditional_losses_832871

inputs!
sequential_7_832824:!
sequential_7_832826:!
sequential_7_832828:!
sequential_7_832830:-
sequential_7_832832: !
sequential_7_832834: .
sequential_7_832836: А"
sequential_7_832838:	А/
sequential_7_832840:АА"
sequential_7_832842:	А/
sequential_7_832844:АА"
sequential_7_832846:	А'
sequential_7_832848:
А@А"
sequential_7_832850:	А"
dense_15_832853:	А
dense_15_832855:
identityИв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpв1dense_14/kernel/Regularizer/Square/ReadVariableOpв dense_15/StatefulPartitionedCallв$sequential_7/StatefulPartitionedCall└
$sequential_7/StatefulPartitionedCallStatefulPartitionedCallinputssequential_7_832824sequential_7_832826sequential_7_832828sequential_7_832830sequential_7_832832sequential_7_832834sequential_7_832836sequential_7_832838sequential_7_832840sequential_7_832842sequential_7_832844sequential_7_832846sequential_7_832848sequential_7_832850*
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
*2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_8324912&
$sequential_7/StatefulPartitionedCall└
 dense_15/StatefulPartitionedCallStatefulPartitionedCall-sequential_7/StatefulPartitionedCall:output:0dense_15_832853dense_15_832855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_8327182"
 dense_15/StatefulPartitionedCall─
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_832832*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╝
1dense_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_7_832848* 
_output_shapes
:
А@А*
dtype023
1dense_14/kernel/Regularizer/Square/ReadVariableOp╕
"dense_14/kernel/Regularizer/SquareSquare9dense_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
А@А2$
"dense_14/kernel/Regularizer/SquareЧ
!dense_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_14/kernel/Regularizer/Const╛
dense_14/kernel/Regularizer/SumSum&dense_14/kernel/Regularizer/Square:y:0*dense_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/SumЛ
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2#
!dense_14/kernel/Regularizer/mul/x└
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0(dense_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_14/kernel/Regularizer/mul░
IdentityIdentity)dense_15/StatefulPartitionedCall:output:03^conv2d_28/kernel/Regularizer/Square/ReadVariableOp2^dense_14/kernel/Regularizer/Square/ReadVariableOp!^dense_15/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2f
1dense_14/kernel/Regularizer/Square/ReadVariableOp1dense_14/kernel/Regularizer/Square/ReadVariableOp2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
▐
M
1__inference_max_pooling2d_31_layer_call_fn_832022

inputs
identityЄ
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
 *2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_8320162
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
б
Б
E__inference_conv2d_30_layer_call_and_return_conditional_losses_832119

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         А2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
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
Ў
e
F__inference_dropout_14_layer_call_and_return_conditional_losses_834371

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╜
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╟
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
┐
│
E__inference_conv2d_28_layer_call_and_return_conditional_losses_834285

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв2conv2d_28/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu╧
2conv2d_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_28/kernel/Regularizer/SquareSquare:conv2d_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_28/kernel/Regularizer/Squareб
"conv2d_28/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_28/kernel/Regularizer/Const┬
 conv2d_28/kernel/Regularizer/SumSum'conv2d_28/kernel/Regularizer/Square:y:0+conv2d_28/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/SumН
"conv2d_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2$
"conv2d_28/kernel/Regularizer/mul/x─
 conv2d_28/kernel/Regularizer/mulMul+conv2d_28/kernel/Regularizer/mul/x:output:0)conv2d_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_28/kernel/Regularizer/mul╘
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_28/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         KK 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_28/kernel/Regularizer/Square/ReadVariableOp2conv2d_28/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs"╠L
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
StatefulPartitionedCall:0         tensorflow/serving/predict:╫Й
Б

h2ptjl
_output
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
щ_default_save_signature
+ъ&call_and_return_all_conditional_losses
ы__call__
	ьcall"П	
_tf_keras_modelї{"name": "CNN3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN3", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN3"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
░m
	layer-0

layer_with_weights-0

layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
layer_with_weights-4
layer-8
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
trainable_variables
	variables
regularization_losses
	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"╔i
_tf_keras_sequentialкi{"name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_7_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_28", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_29", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_30", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_31", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 33, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_7_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_7_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_28", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_29", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_30", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_31", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 26}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 27}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31}, {"class_name": "Dropout", "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 32}]}}}
╫

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
+я&call_and_return_all_conditional_losses
Ё__call__"░
_tf_keras_layerЦ{"name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
ы
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m╧'m╨(m╤)m╥*m╙+m╘,m╒-m╓.m╫/m╪0m┘1m┌v█v▄&v▌'v▐(v▀)vр*vс+vт,vу-vф.vх/vц0vч1vш"
	optimizer
Ж
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
12
13"
trackable_list_wrapper
Ц
&0
'1
22
33
(4
)5
*6
+7
,8
-9
.10
/11
012
113
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
╬
trainable_variables
4non_trainable_variables
	variables
regularization_losses
5layer_regularization_losses
6layer_metrics

7layers
8metrics
ы__call__
щ_default_save_signature
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
-
ёserving_default"
signature_map
╪
9trainable_variables
:	variables
;regularization_losses
<	keras_api
+Є&call_and_return_all_conditional_losses
є__call__"╟
_tf_keras_layerн{"name": "lambda_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_7", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT61AAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
─

=axis
	&gamma
'beta
2moving_mean
3moving_variance
>trainable_variables
?	variables
@regularization_losses
A	keras_api
+Ї&call_and_return_all_conditional_losses
ї__call__"ю
_tf_keras_layer╘{"name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
в

(kernel
)bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
+Ў&call_and_return_all_conditional_losses
ў__call__"√	
_tf_keras_layerс	{"name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
│
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+°&call_and_return_all_conditional_losses
∙__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_28", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 40}}
╓


*kernel
+bias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
+·&call_and_return_all_conditional_losses
√__call__"п	
_tf_keras_layerХ	{"name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
│
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
+№&call_and_return_all_conditional_losses
¤__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_29", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 42}}
╪


,kernel
-bias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
+■&call_and_return_all_conditional_losses
 __call__"▒	
_tf_keras_layerЧ	{"name": "conv2d_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
│
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_30", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 44}}
╓


.kernel
/bias
Ztrainable_variables
[	variables
\regularization_losses
]	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"п	
_tf_keras_layerХ	{"name": "conv2d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 9, 9, 256]}}
│
^trainable_variables
_	variables
`regularization_losses
a	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"в
_tf_keras_layerИ{"name": "max_pooling2d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_31", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 46}}
Б
btrainable_variables
c	variables
dregularization_losses
e	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"Ё
_tf_keras_layer╓{"name": "dropout_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 26}
Ш
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"З
_tf_keras_layerэ{"name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 47}}
и	

0kernel
1bias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"Б
_tf_keras_layerч{"name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 8192]}}
Б
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"Ё
_tf_keras_layer╓{"name": "dropout_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 32}
v
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111"
trackable_list_wrapper
Ж
&0
'1
22
33
(4
)5
*6
+7
,8
-9
.10
/11
012
113"
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
░
trainable_variables
rnon_trainable_variables
	variables
regularization_losses
slayer_regularization_losses
tlayer_metrics

ulayers
vmetrics
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_15/kernel
:2dense_15/bias
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
░
trainable_variables
wnon_trainable_variables
	variables
regularization_losses
xlayer_regularization_losses
ylayer_metrics

zlayers
{metrics
Ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
*:( 2conv2d_28/kernel
: 2conv2d_28/bias
+:) А2conv2d_29/kernel
:А2conv2d_29/bias
,:*АА2conv2d_30/kernel
:А2conv2d_30/bias
,:*АА2conv2d_31/kernel
:А2conv2d_31/bias
#:!
А@А2dense_14/kernel
:А2dense_14/bias
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
│
9trainable_variables
~non_trainable_variables
:	variables
;regularization_losses
layer_regularization_losses
Аlayer_metrics
Бlayers
Вmetrics
є__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
<
&0
'1
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
>trainable_variables
Гnon_trainable_variables
?	variables
@regularization_losses
 Дlayer_regularization_losses
Еlayer_metrics
Жlayers
Зmetrics
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
(
О0"
trackable_list_wrapper
╡
Btrainable_variables
Иnon_trainable_variables
C	variables
Dregularization_losses
 Йlayer_regularization_losses
Кlayer_metrics
Лlayers
Мmetrics
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Ftrainable_variables
Нnon_trainable_variables
G	variables
Hregularization_losses
 Оlayer_regularization_losses
Пlayer_metrics
Рlayers
Сmetrics
∙__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Jtrainable_variables
Тnon_trainable_variables
K	variables
Lregularization_losses
 Уlayer_regularization_losses
Фlayer_metrics
Хlayers
Цmetrics
√__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Ntrainable_variables
Чnon_trainable_variables
O	variables
Pregularization_losses
 Шlayer_regularization_losses
Щlayer_metrics
Ъlayers
Ыmetrics
¤__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Rtrainable_variables
Ьnon_trainable_variables
S	variables
Tregularization_losses
 Эlayer_regularization_losses
Юlayer_metrics
Яlayers
аmetrics
 __call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Vtrainable_variables
бnon_trainable_variables
W	variables
Xregularization_losses
 вlayer_regularization_losses
гlayer_metrics
дlayers
еmetrics
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Ztrainable_variables
жnon_trainable_variables
[	variables
\regularization_losses
 зlayer_regularization_losses
иlayer_metrics
йlayers
кmetrics
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
^trainable_variables
лnon_trainable_variables
_	variables
`regularization_losses
 мlayer_regularization_losses
нlayer_metrics
оlayers
пmetrics
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
btrainable_variables
░non_trainable_variables
c	variables
dregularization_losses
 ▒layer_regularization_losses
▓layer_metrics
│layers
┤metrics
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
ftrainable_variables
╡non_trainable_variables
g	variables
hregularization_losses
 ╢layer_regularization_losses
╖layer_metrics
╕layers
╣metrics
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
(
П0"
trackable_list_wrapper
╡
jtrainable_variables
║non_trainable_variables
k	variables
lregularization_losses
 ╗layer_regularization_losses
╝layer_metrics
╜layers
╛metrics
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
ntrainable_variables
┐non_trainable_variables
o	variables
pregularization_losses
 └layer_regularization_losses
┴layer_metrics
┬layers
├metrics
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ж
	0

1
2
3
4
5
6
7
8
9
10
11
12
13"
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
╪

─total

┼count
╞	variables
╟	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}
Ы

╚total

╔count
╩
_fn_kwargs
╦	variables
╠	keras_api"╧
_tf_keras_metric┤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}
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
.
20
31"
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
(
О0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
П0"
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
:  (2total
:  (2count
0
─0
┼1"
trackable_list_wrapper
.
╞	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╚0
╔1"
trackable_list_wrapper
.
╦	variables"
_generic_user_object
':%	А2Adam/dense_15/kernel/m
 :2Adam/dense_15/bias/m
.:,2"Adam/batch_normalization_7/gamma/m
-:+2!Adam/batch_normalization_7/beta/m
/:- 2Adam/conv2d_28/kernel/m
!: 2Adam/conv2d_28/bias/m
0:. А2Adam/conv2d_29/kernel/m
": А2Adam/conv2d_29/bias/m
1:/АА2Adam/conv2d_30/kernel/m
": А2Adam/conv2d_30/bias/m
1:/АА2Adam/conv2d_31/kernel/m
": А2Adam/conv2d_31/bias/m
(:&
А@А2Adam/dense_14/kernel/m
!:А2Adam/dense_14/bias/m
':%	А2Adam/dense_15/kernel/v
 :2Adam/dense_15/bias/v
.:,2"Adam/batch_normalization_7/gamma/v
-:+2!Adam/batch_normalization_7/beta/v
/:- 2Adam/conv2d_28/kernel/v
!: 2Adam/conv2d_28/bias/v
0:. А2Adam/conv2d_29/kernel/v
": А2Adam/conv2d_29/bias/v
1:/АА2Adam/conv2d_30/kernel/v
": А2Adam/conv2d_30/bias/v
1:/АА2Adam/conv2d_31/kernel/v
": А2Adam/conv2d_31/bias/v
(:&
А@А2Adam/dense_14/kernel/v
!:А2Adam/dense_14/bias/v
ч2ф
!__inference__wrapped_model_831848╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
input_1         KK
┬2┐
@__inference_CNN3_layer_call_and_return_conditional_losses_833184
@__inference_CNN3_layer_call_and_return_conditional_losses_833282
@__inference_CNN3_layer_call_and_return_conditional_losses_833366
@__inference_CNN3_layer_call_and_return_conditional_losses_833464┤
л▓з
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
%__inference_CNN3_layer_call_fn_833501
%__inference_CNN3_layer_call_fn_833538
%__inference_CNN3_layer_call_fn_833575
%__inference_CNN3_layer_call_fn_833612┤
л▓з
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Д2Б
__inference_call_798342
__inference_call_798414
__inference_call_798486│
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
ю2ы
H__inference_sequential_7_layer_call_and_return_conditional_losses_833701
H__inference_sequential_7_layer_call_and_return_conditional_losses_833792
H__inference_sequential_7_layer_call_and_return_conditional_losses_833869
H__inference_sequential_7_layer_call_and_return_conditional_losses_833960└
╖▓│
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
-__inference_sequential_7_layer_call_fn_833993
-__inference_sequential_7_layer_call_fn_834026
-__inference_sequential_7_layer_call_fn_834059
-__inference_sequential_7_layer_call_fn_834092└
╖▓│
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
D__inference_dense_15_layer_call_and_return_conditional_losses_834103в
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
╙2╨
)__inference_dense_15_layer_call_fn_834112в
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
╦B╚
$__inference_signature_wrapper_833100input_1"Ф
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
╥2╧
D__inference_lambda_7_layer_call_and_return_conditional_losses_834120
D__inference_lambda_7_layer_call_and_return_conditional_losses_834128└
╖▓│
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ь2Щ
)__inference_lambda_7_layer_call_fn_834133
)__inference_lambda_7_layer_call_fn_834138└
╖▓│
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ж2Г
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834156
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834174
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834192
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834210┤
л▓з
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_7_layer_call_fn_834223
6__inference_batch_normalization_7_layer_call_fn_834236
6__inference_batch_normalization_7_layer_call_fn_834249
6__inference_batch_normalization_7_layer_call_fn_834262┤
л▓з
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
E__inference_conv2d_28_layer_call_and_return_conditional_losses_834285в
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
╘2╤
*__inference_conv2d_28_layer_call_fn_834294в
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
┤2▒
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_831980р
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
annotationsк *@в=
;К84                                    
Щ2Ц
1__inference_max_pooling2d_28_layer_call_fn_831986р
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
annotationsк *@в=
;К84                                    
я2ь
E__inference_conv2d_29_layer_call_and_return_conditional_losses_834305в
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
╘2╤
*__inference_conv2d_29_layer_call_fn_834314в
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
┤2▒
L__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_831992р
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
annotationsк *@в=
;К84                                    
Щ2Ц
1__inference_max_pooling2d_29_layer_call_fn_831998р
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
annotationsк *@в=
;К84                                    
я2ь
E__inference_conv2d_30_layer_call_and_return_conditional_losses_834325в
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
╘2╤
*__inference_conv2d_30_layer_call_fn_834334в
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
┤2▒
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_832004р
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
annotationsк *@в=
;К84                                    
Щ2Ц
1__inference_max_pooling2d_30_layer_call_fn_832010р
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
annotationsк *@в=
;К84                                    
я2ь
E__inference_conv2d_31_layer_call_and_return_conditional_losses_834345в
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
╘2╤
*__inference_conv2d_31_layer_call_fn_834354в
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
┤2▒
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_832016р
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
annotationsк *@в=
;К84                                    
Щ2Ц
1__inference_max_pooling2d_31_layer_call_fn_832022р
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
annotationsк *@в=
;К84                                    
╩2╟
F__inference_dropout_14_layer_call_and_return_conditional_losses_834359
F__inference_dropout_14_layer_call_and_return_conditional_losses_834371┤
л▓з
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ф2С
+__inference_dropout_14_layer_call_fn_834376
+__inference_dropout_14_layer_call_fn_834381┤
л▓з
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
E__inference_flatten_7_layer_call_and_return_conditional_losses_834387в
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
╘2╤
*__inference_flatten_7_layer_call_fn_834392в
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
ю2ы
D__inference_dense_14_layer_call_and_return_conditional_losses_834415в
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
╙2╨
)__inference_dense_14_layer_call_fn_834424в
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
╩2╟
F__inference_dropout_15_layer_call_and_return_conditional_losses_834429
F__inference_dropout_15_layer_call_and_return_conditional_losses_834441┤
л▓з
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ф2С
+__inference_dropout_15_layer_call_fn_834446
+__inference_dropout_15_layer_call_fn_834451┤
л▓з
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

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
│2░
__inference_loss_fn_0_834462П
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
│2░
__inference_loss_fn_1_834473П
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
annotationsк *в ║
@__inference_CNN3_layer_call_and_return_conditional_losses_833184v&'23()*+,-./01;в8
1в.
(К%
inputs         KK
p 
к "%в"
К
0         
Ъ ║
@__inference_CNN3_layer_call_and_return_conditional_losses_833282v&'23()*+,-./01;в8
1в.
(К%
inputs         KK
p
к "%в"
К
0         
Ъ ╗
@__inference_CNN3_layer_call_and_return_conditional_losses_833366w&'23()*+,-./01<в9
2в/
)К&
input_1         KK
p 
к "%в"
К
0         
Ъ ╗
@__inference_CNN3_layer_call_and_return_conditional_losses_833464w&'23()*+,-./01<в9
2в/
)К&
input_1         KK
p
к "%в"
К
0         
Ъ У
%__inference_CNN3_layer_call_fn_833501j&'23()*+,-./01<в9
2в/
)К&
input_1         KK
p 
к "К         Т
%__inference_CNN3_layer_call_fn_833538i&'23()*+,-./01;в8
1в.
(К%
inputs         KK
p 
к "К         Т
%__inference_CNN3_layer_call_fn_833575i&'23()*+,-./01;в8
1в.
(К%
inputs         KK
p
к "К         У
%__inference_CNN3_layer_call_fn_833612j&'23()*+,-./01<в9
2в/
)К&
input_1         KK
p
к "К         з
!__inference__wrapped_model_831848Б&'23()*+,-./018в5
.в+
)К&
input_1         KK
к "3к0
.
output_1"К
output_1         ь
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834156Ц&'23MвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ь
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834174Ц&'23MвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ╟
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834192r&'23;в8
1в.
(К%
inputs         KK
p 
к "-в*
#К 
0         KK
Ъ ╟
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_834210r&'23;в8
1в.
(К%
inputs         KK
p
к "-в*
#К 
0         KK
Ъ ─
6__inference_batch_normalization_7_layer_call_fn_834223Й&'23MвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           ─
6__inference_batch_normalization_7_layer_call_fn_834236Й&'23MвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           Я
6__inference_batch_normalization_7_layer_call_fn_834249e&'23;в8
1в.
(К%
inputs         KK
p 
к " К         KKЯ
6__inference_batch_normalization_7_layer_call_fn_834262e&'23;в8
1в.
(К%
inputs         KK
p
к " К         KKt
__inference_call_798342Y&'23()*+,-./013в0
)в&
 К
inputsАKK
p
к "К	Аt
__inference_call_798414Y&'23()*+,-./013в0
)в&
 К
inputsАKK
p 
к "К	АД
__inference_call_798486i&'23()*+,-./01;в8
1в.
(К%
inputs         KK
p 
к "К         ╡
E__inference_conv2d_28_layer_call_and_return_conditional_losses_834285l()7в4
-в*
(К%
inputs         KK
к "-в*
#К 
0         KK 
Ъ Н
*__inference_conv2d_28_layer_call_fn_834294_()7в4
-в*
(К%
inputs         KK
к " К         KK ╢
E__inference_conv2d_29_layer_call_and_return_conditional_losses_834305m*+7в4
-в*
(К%
inputs         %% 
к ".в+
$К!
0         %%А
Ъ О
*__inference_conv2d_29_layer_call_fn_834314`*+7в4
-в*
(К%
inputs         %% 
к "!К         %%А╖
E__inference_conv2d_30_layer_call_and_return_conditional_losses_834325n,-8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ П
*__inference_conv2d_30_layer_call_fn_834334a,-8в5
.в+
)К&
inputs         А
к "!К         А╖
E__inference_conv2d_31_layer_call_and_return_conditional_losses_834345n./8в5
.в+
)К&
inputs         		А
к ".в+
$К!
0         		А
Ъ П
*__inference_conv2d_31_layer_call_fn_834354a./8в5
.в+
)К&
inputs         		А
к "!К         		Аж
D__inference_dense_14_layer_call_and_return_conditional_losses_834415^010в-
&в#
!К
inputs         А@
к "&в#
К
0         А
Ъ ~
)__inference_dense_14_layer_call_fn_834424Q010в-
&в#
!К
inputs         А@
к "К         Ае
D__inference_dense_15_layer_call_and_return_conditional_losses_834103]0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ }
)__inference_dense_15_layer_call_fn_834112P0в-
&в#
!К
inputs         А
к "К         ╕
F__inference_dropout_14_layer_call_and_return_conditional_losses_834359n<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ ╕
F__inference_dropout_14_layer_call_and_return_conditional_losses_834371n<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ Р
+__inference_dropout_14_layer_call_fn_834376a<в9
2в/
)К&
inputs         А
p 
к "!К         АР
+__inference_dropout_14_layer_call_fn_834381a<в9
2в/
)К&
inputs         А
p
к "!К         Аи
F__inference_dropout_15_layer_call_and_return_conditional_losses_834429^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ и
F__inference_dropout_15_layer_call_and_return_conditional_losses_834441^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ А
+__inference_dropout_15_layer_call_fn_834446Q4в1
*в'
!К
inputs         А
p 
к "К         АА
+__inference_dropout_15_layer_call_fn_834451Q4в1
*в'
!К
inputs         А
p
к "К         Ал
E__inference_flatten_7_layer_call_and_return_conditional_losses_834387b8в5
.в+
)К&
inputs         А
к "&в#
К
0         А@
Ъ Г
*__inference_flatten_7_layer_call_fn_834392U8в5
.в+
)К&
inputs         А
к "К         А@╕
D__inference_lambda_7_layer_call_and_return_conditional_losses_834120p?в<
5в2
(К%
inputs         KK

 
p 
к "-в*
#К 
0         KK
Ъ ╕
D__inference_lambda_7_layer_call_and_return_conditional_losses_834128p?в<
5в2
(К%
inputs         KK

 
p
к "-в*
#К 
0         KK
Ъ Р
)__inference_lambda_7_layer_call_fn_834133c?в<
5в2
(К%
inputs         KK

 
p 
к " К         KKР
)__inference_lambda_7_layer_call_fn_834138c?в<
5в2
(К%
inputs         KK

 
p
к " К         KK;
__inference_loss_fn_0_834462(в

в 
к "К ;
__inference_loss_fn_1_8344730в

в 
к "К я
L__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_831980ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_28_layer_call_fn_831986СRвO
HвE
CК@
inputs4                                    
к ";К84                                    я
L__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_831992ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_29_layer_call_fn_831998СRвO
HвE
CК@
inputs4                                    
к ";К84                                    я
L__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_832004ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_30_layer_call_fn_832010СRвO
HвE
CК@
inputs4                                    
к ";К84                                    я
L__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_832016ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╟
1__inference_max_pooling2d_31_layer_call_fn_832022СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ┼
H__inference_sequential_7_layer_call_and_return_conditional_losses_833701y&'23()*+,-./01?в<
5в2
(К%
inputs         KK
p 

 
к "&в#
К
0         А
Ъ ┼
H__inference_sequential_7_layer_call_and_return_conditional_losses_833792y&'23()*+,-./01?в<
5в2
(К%
inputs         KK
p

 
к "&в#
К
0         А
Ъ ╬
H__inference_sequential_7_layer_call_and_return_conditional_losses_833869Б&'23()*+,-./01GвD
=в:
0К-
lambda_7_input         KK
p 

 
к "&в#
К
0         А
Ъ ╬
H__inference_sequential_7_layer_call_and_return_conditional_losses_833960Б&'23()*+,-./01GвD
=в:
0К-
lambda_7_input         KK
p

 
к "&в#
К
0         А
Ъ е
-__inference_sequential_7_layer_call_fn_833993t&'23()*+,-./01GвD
=в:
0К-
lambda_7_input         KK
p 

 
к "К         АЭ
-__inference_sequential_7_layer_call_fn_834026l&'23()*+,-./01?в<
5в2
(К%
inputs         KK
p 

 
к "К         АЭ
-__inference_sequential_7_layer_call_fn_834059l&'23()*+,-./01?в<
5в2
(К%
inputs         KK
p

 
к "К         Ае
-__inference_sequential_7_layer_call_fn_834092t&'23()*+,-./01GвD
=в:
0К-
lambda_7_input         KK
p

 
к "К         А╡
$__inference_signature_wrapper_833100М&'23()*+,-./01Cв@
в 
9к6
4
input_1)К&
input_1         KK"3к0
.
output_1"К
output_1         