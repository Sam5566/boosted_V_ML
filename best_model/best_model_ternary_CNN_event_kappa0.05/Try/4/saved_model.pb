»╗
С┤
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
Џ
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
Щ
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
epsilonfloat%иЛ8"&
exponential_avg_factorfloat%  ђ?";
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
ѓ
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
delete_old_dirsbool(ѕ
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ую
{
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ* 
shared_namedense_14/kernel
t
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes
:	ђ*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
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
ј
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma
Є
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
ї
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
Ё
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
џ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean
Њ
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
б
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance
Џ
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
ё
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
: *
dtype0
Ё
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*!
shared_nameconv2d_13/kernel
~
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*'
_output_shapes
: ђ*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_14/kernel

$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_14/bias
n
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes	
:ђ*
dtype0
}
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђбђ* 
shared_namedense_12/kernel
v
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*!
_output_shapes
:ђбђ*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:ђ*
dtype0
|
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ* 
shared_namedense_13/kernel
u
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel* 
_output_shapes
:
ђђ*
dtype0
s
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_13/bias
l
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes	
:ђ*
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
Ѕ
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_14/kernel/m
ѓ
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes
:	ђ*
dtype0
ђ
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/m
Ћ
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/m
Њ
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_12/kernel/m
І
+Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/m*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_12/bias/m
{
)Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/m*
_output_shapes
: *
dtype0
Њ
Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*(
shared_nameAdam/conv2d_13/kernel/m
ї
+Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/m*'
_output_shapes
: ђ*
dtype0
Ѓ
Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_13/bias/m
|
)Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_14/kernel/m
Ї
+Adam/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_14/bias/m
|
)Adam/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/m*
_output_shapes	
:ђ*
dtype0
І
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђбђ*'
shared_nameAdam/dense_12/kernel/m
ё
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*!
_output_shapes
:ђбђ*
dtype0
Ђ
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_12/bias/m
z
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_13/kernel/m
Ѓ
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_13/bias/m
z
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_14/kernel/v
ѓ
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes
:	ђ*
dtype0
ђ
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0
ю
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/v
Ћ
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
:*
dtype0
џ
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/v
Њ
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
:*
dtype0
њ
Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_12/kernel/v
І
+Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/v*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_12/bias/v
{
)Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/v*
_output_shapes
: *
dtype0
Њ
Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: ђ*(
shared_nameAdam/conv2d_13/kernel/v
ї
+Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/v*'
_output_shapes
: ђ*
dtype0
Ѓ
Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_13/bias/v
|
)Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_14/kernel/v
Ї
+Adam/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_14/bias/v
|
)Adam/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/v*
_output_shapes	
:ђ*
dtype0
І
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђбђ*'
shared_nameAdam/dense_12/kernel/v
ё
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*!
_output_shapes
:ђбђ*
dtype0
Ђ
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_12/bias/v
z
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes	
:ђ*
dtype0
і
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*'
shared_nameAdam/dense_13/kernel/v
Ѓ
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v* 
_output_shapes
:
ђђ*
dtype0
Ђ
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_13/bias/v
z
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes	
:ђ*
dtype0

NoOpNoOp
У]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Б]
valueЎ]Bќ] BЈ]
і

h2ptjl
_output
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
е
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
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
п
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m¤'mл*mЛ+mм,mМ-mн.mН/mо0mО1mп2m┘3m┌v█v▄&vП'vя*v▀+vЯ,vр-vР.vс/vС0vт1vТ2vу3vУ
 
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
111
212
313
14
15
f
&0
'1
*2
+3
,4
-5
.6
/7
08
19
210
311
12
13
Г
regularization_losses
4non_trainable_variables
	variables
5layer_regularization_losses
6metrics

7layers
8layer_metrics
trainable_variables
 
R
9regularization_losses
:	variables
;trainable_variables
<	keras_api
Ќ
=axis
	&gamma
'beta
(moving_mean
)moving_variance
>regularization_losses
?	variables
@trainable_variables
A	keras_api
h

*kernel
+bias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
R
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

,kernel
-bias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
h

.kernel
/bias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
R
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
R
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
R
^regularization_losses
_	variables
`trainable_variables
a	keras_api
h

0kernel
1bias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
R
fregularization_losses
g	variables
htrainable_variables
i	keras_api
h

2kernel
3bias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
R
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
 
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
212
313
V
&0
'1
*2
+3
,4
-5
.6
/7
08
19
210
311
Г
regularization_losses
rnon_trainable_variables
	variables
slayer_regularization_losses
tmetrics

ulayers
vlayer_metrics
trainable_variables
NL
VARIABLE_VALUEdense_14/kernel)_output/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_14/bias'_output/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
wnon_trainable_variables
regularization_losses
xlayer_regularization_losses
	variables
ymetrics

zlayers
{layer_metrics
trainable_variables
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
WU
VARIABLE_VALUEbatch_normalization_4/gamma&variables/0/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_4/beta&variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_4/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_4/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_12/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_12/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_13/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_13/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_14/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_14/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_12/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_12/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_13/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_13/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

|0
}1

0
1
 
 
 
 
░
~non_trainable_variables
9regularization_losses
layer_regularization_losses
:	variables
ђmetrics
Ђlayers
ѓlayer_metrics
;trainable_variables
 
 

&0
'1
(2
)3

&0
'1
▓
Ѓnon_trainable_variables
>regularization_losses
 ёlayer_regularization_losses
?	variables
Ёmetrics
єlayers
Єlayer_metrics
@trainable_variables
 

*0
+1

*0
+1
▓
ѕnon_trainable_variables
Bregularization_losses
 Ѕlayer_regularization_losses
C	variables
іmetrics
Іlayers
їlayer_metrics
Dtrainable_variables
 
 
 
▓
Їnon_trainable_variables
Fregularization_losses
 јlayer_regularization_losses
G	variables
Јmetrics
љlayers
Љlayer_metrics
Htrainable_variables
 

,0
-1

,0
-1
▓
њnon_trainable_variables
Jregularization_losses
 Њlayer_regularization_losses
K	variables
ћmetrics
Ћlayers
ќlayer_metrics
Ltrainable_variables
 
 
 
▓
Ќnon_trainable_variables
Nregularization_losses
 ўlayer_regularization_losses
O	variables
Ўmetrics
џlayers
Џlayer_metrics
Ptrainable_variables
 

.0
/1

.0
/1
▓
юnon_trainable_variables
Rregularization_losses
 Юlayer_regularization_losses
S	variables
ъmetrics
Ъlayers
аlayer_metrics
Ttrainable_variables
 
 
 
▓
Аnon_trainable_variables
Vregularization_losses
 бlayer_regularization_losses
W	variables
Бmetrics
цlayers
Цlayer_metrics
Xtrainable_variables
 
 
 
▓
дnon_trainable_variables
Zregularization_losses
 Дlayer_regularization_losses
[	variables
еmetrics
Еlayers
фlayer_metrics
\trainable_variables
 
 
 
▓
Фnon_trainable_variables
^regularization_losses
 гlayer_regularization_losses
_	variables
Гmetrics
«layers
»layer_metrics
`trainable_variables
 

00
11

00
11
▓
░non_trainable_variables
bregularization_losses
 ▒layer_regularization_losses
c	variables
▓metrics
│layers
┤layer_metrics
dtrainable_variables
 
 
 
▓
хnon_trainable_variables
fregularization_losses
 Хlayer_regularization_losses
g	variables
иmetrics
Иlayers
╣layer_metrics
htrainable_variables
 

20
31

20
31
▓
║non_trainable_variables
jregularization_losses
 ╗layer_regularization_losses
k	variables
╝metrics
йlayers
Йlayer_metrics
ltrainable_variables
 
 
 
▓
┐non_trainable_variables
nregularization_losses
 └layer_regularization_losses
o	variables
┴metrics
┬layers
├layer_metrics
ptrainable_variables

(0
)1
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
к	variables
К	keras_api
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
(0
)1
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
к	variables
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
VARIABLE_VALUEAdam/dense_14/kernel/mE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_14/bias/mC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_12/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_12/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_13/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_13/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_14/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_14/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_12/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_12/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_13/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_13/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/dense_14/kernel/vE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_14/bias/vC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_12/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_12/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_13/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_13/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_14/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_14/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_12/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_12/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_13/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_13/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
і
serving_default_input_1Placeholder*/
_output_shapes
:         KK*
dtype0*$
shape:         KK
а
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/bias*
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
GPU2 *0J 8ѓ *-
f(R&
$__inference_signature_wrapper_703250
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp+Adam/conv2d_12/kernel/m/Read/ReadVariableOp)Adam/conv2d_12/bias/m/Read/ReadVariableOp+Adam/conv2d_13/kernel/m/Read/ReadVariableOp)Adam/conv2d_13/bias/m/Read/ReadVariableOp+Adam/conv2d_14/kernel/m/Read/ReadVariableOp)Adam/conv2d_14/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp+Adam/conv2d_12/kernel/v/Read/ReadVariableOp)Adam/conv2d_12/bias/v/Read/ReadVariableOp+Adam/conv2d_13/kernel/v/Read/ReadVariableOp)Adam/conv2d_13/bias/v/Read/ReadVariableOp+Adam/conv2d_14/kernel/v/Read/ReadVariableOp)Adam/conv2d_14/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpConst*B
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
GPU2 *0J 8ѓ *(
f#R!
__inference__traced_save_704937
└
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biastotalcounttotal_1count_1Adam/dense_14/kernel/mAdam/dense_14/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/conv2d_12/kernel/mAdam/conv2d_12/bias/mAdam/conv2d_13/kernel/mAdam/conv2d_13/bias/mAdam/conv2d_14/kernel/mAdam/conv2d_14/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/conv2d_12/kernel/vAdam/conv2d_12/bias/vAdam/conv2d_13/kernel/vAdam/conv2d_13/bias/vAdam/conv2d_14/kernel/vAdam/conv2d_14/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v*A
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
GPU2 *0J 8ѓ *+
f&R$
"__inference__traced_restore_705106ьГ
Б
Ќ
)__inference_dense_14_layer_call_fn_704344

inputs
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCallщ
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_7028382
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Л
б
*__inference_conv2d_14_layer_call_fn_704566

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_7021802
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
┐
Г
D__inference_dense_12_layer_call_and_return_conditional_losses_704627

inputs3
matmul_readvariableop_resource:ђбђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpљ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Relu╚
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђб: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         ђб
 
_user_specified_nameinputs
╬
А
*__inference_conv2d_13_layer_call_fn_704546

inputs"
unknown: ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_7021622
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         %%ђ2

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
х
e
F__inference_dropout_14_layer_call_and_return_conditional_losses_704712

inputs
identityѕc
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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
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
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
н
§
-__inference_sequential_4_layer_call_fn_704258

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ
identityѕбStatefulPartitionedCallЮ
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
:         ђ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_7022812
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

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
┐
│
E__inference_conv2d_12_layer_call_and_return_conditional_losses_704517

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu¤
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulн
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
џ
╗
__inference_loss_fn_0_704733U
;conv2d_12_kernel_regularizer_square_readvariableop_resource: 
identityѕб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpВ
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_12_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulю
IdentityIdentity$conv2d_12/kernel/Regularizer/mul:z:03^conv2d_12/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp
Ю
ђ
E__inference_conv2d_13_layer_call_and_return_conditional_losses_704537

inputs9
conv2d_readvariableop_resource: ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%ђ2

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
э
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_704641

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш┴
б
?__inference_CNN_layer_call_and_return_conditional_losses_703451

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_12_conv2d_readvariableop_resource: D
6sequential_4_conv2d_12_biasadd_readvariableop_resource: P
5sequential_4_conv2d_13_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_13_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_14_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_14_biasadd_readvariableop_resource:	ђI
4sequential_4_dense_12_matmul_readvariableop_resource:ђбђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpб1sequential_4/batch_normalization_4/AssignNewValueб3sequential_4/batch_normalization_4/AssignNewValue_1бBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_12/BiasAdd/ReadVariableOpб,sequential_4/conv2d_12/Conv2D/ReadVariableOpб-sequential_4/conv2d_13/BiasAdd/ReadVariableOpб,sequential_4/conv2d_13/Conv2D/ReadVariableOpб-sequential_4/conv2d_14/BiasAdd/ReadVariableOpб,sequential_4/conv2d_14/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2в
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1л
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<25
3sequential_4/batch_normalization_4/FusedBatchNormV3ы
1sequential_4/batch_normalization_4/AssignNewValueAssignVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource@sequential_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0C^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_4/batch_normalization_4/AssignNewValue§
3sequential_4/batch_normalization_4/AssignNewValue_1AssignVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceDsequential_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0E^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_4/batch_normalization_4/AssignNewValue_1┌
,sequential_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_12/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_12/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_12/Conv2DЛ
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpС
sequential_4/conv2d_12/BiasAddBiasAdd&sequential_4/conv2d_12/Conv2D:output:05sequential_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_12/BiasAddЦ
sequential_4/conv2d_12/ReluRelu'sequential_4/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_12/Reluы
%sequential_4/max_pooling2d_12/MaxPoolMaxPool)sequential_4/conv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_12/MaxPool█
,sequential_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_13/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_13/Conv2DConv2D.sequential_4/max_pooling2d_12/MaxPool:output:04sequential_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_13/Conv2Dм
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpт
sequential_4/conv2d_13/BiasAddBiasAdd&sequential_4/conv2d_13/Conv2D:output:05sequential_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_13/BiasAddд
sequential_4/conv2d_13/ReluRelu'sequential_4/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_13/ReluЫ
%sequential_4/max_pooling2d_13/MaxPoolMaxPool)sequential_4/conv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_13/MaxPool▄
,sequential_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_14/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_14/Conv2DConv2D.sequential_4/max_pooling2d_13/MaxPool:output:04sequential_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_14/Conv2Dм
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpт
sequential_4/conv2d_14/BiasAddBiasAdd&sequential_4/conv2d_14/Conv2D:output:05sequential_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_14/BiasAddд
sequential_4/conv2d_14/ReluRelu'sequential_4/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_14/ReluЫ
%sequential_4/max_pooling2d_14/MaxPoolMaxPool)sequential_4/conv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_14/MaxPoolЊ
%sequential_4/dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2'
%sequential_4/dropout_12/dropout/ConstВ
#sequential_4/dropout_12/dropout/MulMul.sequential_4/max_pooling2d_14/MaxPool:output:0.sequential_4/dropout_12/dropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2%
#sequential_4/dropout_12/dropout/Mulг
%sequential_4/dropout_12/dropout/ShapeShape.sequential_4/max_pooling2d_14/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_12/dropout/ShapeЁ
<sequential_4/dropout_12/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_12/dropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype02>
<sequential_4/dropout_12/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_4/dropout_12/dropout/GreaterEqual/yД
,sequential_4/dropout_12/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_12/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_12/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2.
,sequential_4/dropout_12/dropout/GreaterEqualл
$sequential_4/dropout_12/dropout/CastCast0sequential_4/dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2&
$sequential_4/dropout_12/dropout/Castс
%sequential_4/dropout_12/dropout/Mul_1Mul'sequential_4/dropout_12/dropout/Mul:z:0(sequential_4/dropout_12/dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2'
%sequential_4/dropout_12/dropout/Mul_1Ї
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_4/flatten_4/ConstЛ
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/dropout/Mul_1:z:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_4/flatten_4/Reshapeм
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluЊ
%sequential_4/dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_4/dropout_13/dropout/Constя
#sequential_4/dropout_13/dropout/MulMul(sequential_4/dense_12/Relu:activations:0.sequential_4/dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_4/dropout_13/dropout/Mulд
%sequential_4/dropout_13/dropout/ShapeShape(sequential_4/dense_12/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_13/dropout/Shape§
<sequential_4/dropout_13/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_13/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_4/dropout_13/dropout/GreaterEqual/yЪ
,sequential_4/dropout_13/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_13/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_4/dropout_13/dropout/GreaterEqual╚
$sequential_4/dropout_13/dropout/CastCast0sequential_4/dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_4/dropout_13/dropout/Cast█
%sequential_4/dropout_13/dropout/Mul_1Mul'sequential_4/dropout_13/dropout/Mul:z:0(sequential_4/dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_4/dropout_13/dropout/Mul_1Л
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/dropout/Mul_1:z:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluЊ
%sequential_4/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_4/dropout_14/dropout/Constя
#sequential_4/dropout_14/dropout/MulMul(sequential_4/dense_13/Relu:activations:0.sequential_4/dropout_14/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_4/dropout_14/dropout/Mulд
%sequential_4/dropout_14/dropout/ShapeShape(sequential_4/dense_13/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_14/dropout/Shape§
<sequential_4/dropout_14/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_14/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_14/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_4/dropout_14/dropout/GreaterEqual/yЪ
,sequential_4/dropout_14/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_14/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_4/dropout_14/dropout/GreaterEqual╚
$sequential_4/dropout_14/dropout/CastCast0sequential_4/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_4/dropout_14/dropout/Cast█
%sequential_4/dropout_14/dropout/Mul_1Mul'sequential_4/dropout_14/dropout/Mul:z:0(sequential_4/dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_4/dropout_14/dropout/Mul_1Е
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/SoftmaxТ
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulя
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulП
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulЁ	
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_12/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^sequential_4/batch_normalization_4/AssignNewValue4^sequential_4/batch_normalization_4/AssignNewValue_1C^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_12/BiasAdd/ReadVariableOp-^sequential_4/conv2d_12/Conv2D/ReadVariableOp.^sequential_4/conv2d_13/BiasAdd/ReadVariableOp-^sequential_4/conv2d_13/Conv2D/ReadVariableOp.^sequential_4/conv2d_14/BiasAdd/ReadVariableOp-^sequential_4/conv2d_14/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1sequential_4/batch_normalization_4/AssignNewValue1sequential_4/batch_normalization_4/AssignNewValue2j
3sequential_4/batch_normalization_4/AssignNewValue_13sequential_4/batch_normalization_4/AssignNewValue_12ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp-sequential_4/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_12/Conv2D/ReadVariableOp,sequential_4/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp-sequential_4/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_13/Conv2D/ReadVariableOp,sequential_4/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp-sequential_4/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_14/Conv2D/ReadVariableOp,sequential_4/conv2d_14/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
з1
ф
?__inference_CNN_layer_call_and_return_conditional_losses_703003

inputs!
sequential_4_702950:!
sequential_4_702952:!
sequential_4_702954:!
sequential_4_702956:-
sequential_4_702958: !
sequential_4_702960: .
sequential_4_702962: ђ"
sequential_4_702964:	ђ/
sequential_4_702966:ђђ"
sequential_4_702968:	ђ(
sequential_4_702970:ђбђ"
sequential_4_702972:	ђ'
sequential_4_702974:
ђђ"
sequential_4_702976:	ђ"
dense_14_702979:	ђ
dense_14_702981:
identityѕб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpб dense_14/StatefulPartitionedCallб$sequential_4/StatefulPartitionedCall└
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_702950sequential_4_702952sequential_4_702954sequential_4_702956sequential_4_702958sequential_4_702960sequential_4_702962sequential_4_702964sequential_4_702966sequential_4_702968sequential_4_702970sequential_4_702972sequential_4_702974sequential_4_702976*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_7025992&
$sequential_4/StatefulPartitionedCall└
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0dense_14_702979dense_14_702981*
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_7028382"
 dense_14/StatefulPartitionedCall─
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_702958*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulй
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_702970*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul╝
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_702974* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulС
IdentityIdentity)dense_14/StatefulPartitionedCall:output:03^conv2d_12/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ф
џ
)__inference_dense_12_layer_call_fn_704636

inputs
unknown:ђбђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_7022192
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђб: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         ђб
 
_user_specified_nameinputs
Лџ
Ш
H__inference_sequential_4_layer_call_and_return_conditional_losses_704192
lambda_4_input;
-batch_normalization_4_readvariableop_resource:=
/batch_normalization_4_readvariableop_1_resource:L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: C
(conv2d_13_conv2d_readvariableop_resource: ђ8
)conv2d_13_biasadd_readvariableop_resource:	ђD
(conv2d_14_conv2d_readvariableop_resource:ђђ8
)conv2d_14_biasadd_readvariableop_resource:	ђ<
'dense_12_matmul_readvariableop_resource:ђбђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identityѕб$batch_normalization_4/AssignNewValueб&batch_normalization_4/AssignNewValue_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б conv2d_12/BiasAdd/ReadVariableOpбconv2d_12/Conv2D/ReadVariableOpб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб conv2d_13/BiasAdd/ReadVariableOpбconv2d_13/Conv2D/ReadVariableOpб conv2d_14/BiasAdd/ReadVariableOpбconv2d_14/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stackЎ
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1Ў
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2▓
lambda_4/strided_sliceStridedSlicelambda_4_input%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_4/strided_sliceХ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_4/FusedBatchNormV3░
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue╝
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1│
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOpт
conv2d_12/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_12/Conv2Dф
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp░
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_12/Relu╩
max_pooling2d_12/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPool┤
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_13/Conv2D/ReadVariableOpП
conv2d_13/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_13/Conv2DФ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp▒
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_13/BiasAdd
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_13/Relu╦
max_pooling2d_13/MaxPoolMaxPoolconv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPoolх
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_14/Conv2D/ReadVariableOpП
conv2d_14/Conv2DConv2D!max_pooling2d_13/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_14/Conv2DФ
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp▒
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_14/Relu╦
max_pooling2d_14/MaxPoolMaxPoolconv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_14/MaxPooly
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_12/dropout/ConstИ
dropout_12/dropout/MulMul!max_pooling2d_14/MaxPool:output:0!dropout_12/dropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2
dropout_12/dropout/MulЁ
dropout_12/dropout/ShapeShape!max_pooling2d_14/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shapeя
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype021
/dropout_12/dropout/random_uniform/RandomUniformІ
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_12/dropout/GreaterEqual/yз
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2!
dropout_12/dropout/GreaterEqualЕ
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2
dropout_12/dropout/Cast»
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2
dropout_12/dropout/Mul_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_4/ConstЮ
flatten_4/ReshapeReshapedropout_12/dropout/Mul_1:z:0flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2
flatten_4/ReshapeФ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/MatMulе
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_12/BiasAdd/ReadVariableOpд
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_12/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Constф
dropout_13/dropout/MulMuldense_12/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shapeо
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_13/dropout/random_uniform/RandomUniformІ
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/yв
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_13/dropout/GreaterEqualА
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_13/dropout/CastД
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_13/dropout/Mul_1ф
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_13/MatMul/ReadVariableOpЦ
dense_13/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/MatMulе
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_13/BiasAdd/ReadVariableOpд
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_13/Reluy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_14/dropout/Constф
dropout_14/dropout/MulMuldense_13/Relu:activations:0!dropout_14/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_14/dropout/Mul
dropout_14/dropout/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shapeо
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_14/dropout/random_uniform/RandomUniformІ
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_14/dropout/GreaterEqual/yв
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_14/dropout/GreaterEqualА
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_14/dropout/CastД
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_14/dropout/Mul_1┘
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulЛ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulл
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulш
IdentityIdentitydropout_14/dropout/Mul_1:z:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_4_input
├
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_702117

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
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
epsilon%oЃ:*
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
в
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_702200

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђб2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђб2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
┬
`
D__inference_lambda_4_layer_call_and_return_conditional_losses_704352

inputs
identityЃ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЄ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1Є
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2§
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
И

Ш
D__inference_dense_14_layer_call_and_return_conditional_losses_702838

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ђЋ
╣
?__inference_CNN_layer_call_and_return_conditional_losses_703541
input_1H
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_12_conv2d_readvariableop_resource: D
6sequential_4_conv2d_12_biasadd_readvariableop_resource: P
5sequential_4_conv2d_13_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_13_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_14_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_14_biasadd_readvariableop_resource:	ђI
4sequential_4_dense_12_matmul_readvariableop_resource:ђбђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_12/BiasAdd/ReadVariableOpб,sequential_4/conv2d_12/Conv2D/ReadVariableOpб-sequential_4/conv2d_13/BiasAdd/ReadVariableOpб,sequential_4/conv2d_13/Conv2D/ReadVariableOpб-sequential_4/conv2d_14/BiasAdd/ReadVariableOpб,sequential_4/conv2d_14/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2В
#sequential_4/lambda_4/strided_sliceStridedSliceinput_12sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_12/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_12/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_12/Conv2DЛ
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpС
sequential_4/conv2d_12/BiasAddBiasAdd&sequential_4/conv2d_12/Conv2D:output:05sequential_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_12/BiasAddЦ
sequential_4/conv2d_12/ReluRelu'sequential_4/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_12/Reluы
%sequential_4/max_pooling2d_12/MaxPoolMaxPool)sequential_4/conv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_12/MaxPool█
,sequential_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_13/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_13/Conv2DConv2D.sequential_4/max_pooling2d_12/MaxPool:output:04sequential_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_13/Conv2Dм
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpт
sequential_4/conv2d_13/BiasAddBiasAdd&sequential_4/conv2d_13/Conv2D:output:05sequential_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_13/BiasAddд
sequential_4/conv2d_13/ReluRelu'sequential_4/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_13/ReluЫ
%sequential_4/max_pooling2d_13/MaxPoolMaxPool)sequential_4/conv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_13/MaxPool▄
,sequential_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_14/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_14/Conv2DConv2D.sequential_4/max_pooling2d_13/MaxPool:output:04sequential_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_14/Conv2Dм
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpт
sequential_4/conv2d_14/BiasAddBiasAdd&sequential_4/conv2d_14/Conv2D:output:05sequential_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_14/BiasAddд
sequential_4/conv2d_14/ReluRelu'sequential_4/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_14/ReluЫ
%sequential_4/max_pooling2d_14/MaxPoolMaxPool)sequential_4/conv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_14/MaxPool╗
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_14/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_4/flatten_4/ConstЛ
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_4/flatten_4/Reshapeм
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluГ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluГ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/SoftmaxТ
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulя
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulП
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulЏ
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_12/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_12/BiasAdd/ReadVariableOp-^sequential_4/conv2d_12/Conv2D/ReadVariableOp.^sequential_4/conv2d_13/BiasAdd/ReadVariableOp-^sequential_4/conv2d_13/Conv2D/ReadVariableOp.^sequential_4/conv2d_14/BiasAdd/ReadVariableOp-^sequential_4/conv2d_14/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp-sequential_4/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_12/Conv2D/ReadVariableOp,sequential_4/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp-sequential_4/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_13/Conv2D/ReadVariableOp,sequential_4/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp-sequential_4/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_14/Conv2D/ReadVariableOp,sequential_4/conv2d_14/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
љh
К
__inference__traced_save_704937
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop6
2savev2_adam_conv2d_12_kernel_m_read_readvariableop4
0savev2_adam_conv2d_12_bias_m_read_readvariableop6
2savev2_adam_conv2d_13_kernel_m_read_readvariableop4
0savev2_adam_conv2d_13_bias_m_read_readvariableop6
2savev2_adam_conv2d_14_kernel_m_read_readvariableop4
0savev2_adam_conv2d_14_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop6
2savev2_adam_conv2d_12_kernel_v_read_readvariableop4
0savev2_adam_conv2d_12_bias_v_read_readvariableop6
2savev2_adam_conv2d_13_kernel_v_read_readvariableop4
0savev2_adam_conv2d_13_bias_v_read_readvariableop6
2savev2_adam_conv2d_14_kernel_v_read_readvariableop4
0savev2_adam_conv2d_14_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameТ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*Э
valueЬBв6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЗ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesШ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop2savev2_adam_conv2d_12_kernel_m_read_readvariableop0savev2_adam_conv2d_12_bias_m_read_readvariableop2savev2_adam_conv2d_13_kernel_m_read_readvariableop0savev2_adam_conv2d_13_bias_m_read_readvariableop2savev2_adam_conv2d_14_kernel_m_read_readvariableop0savev2_adam_conv2d_14_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop2savev2_adam_conv2d_12_kernel_v_read_readvariableop0savev2_adam_conv2d_12_bias_v_read_readvariableop2savev2_adam_conv2d_13_kernel_v_read_readvariableop0savev2_adam_conv2d_13_bias_v_read_readvariableop2savev2_adam_conv2d_14_kernel_v_read_readvariableop0savev2_adam_conv2d_14_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*Ж
_input_shapesп
Н: :	ђ:: : : : : ::::: : : ђ:ђ:ђђ:ђ:ђбђ:ђ:
ђђ:ђ: : : : :	ђ:::: : : ђ:ђ:ђђ:ђ:ђбђ:ђ:
ђђ:ђ:	ђ:::: : : ђ:ђ:ђђ:ђ:ђбђ:ђ:
ђђ:ђ: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ђ: 
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
:: 


_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :-)
'
_output_shapes
: ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:'#
!
_output_shapes
:ђбђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:
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
:	ђ: 
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
: ђ:!!

_output_shapes	
:ђ:."*
(
_output_shapes
:ђђ:!#

_output_shapes	
:ђ:'$#
!
_output_shapes
:ђбђ:!%

_output_shapes	
:ђ:&&"
 
_output_shapes
:
ђђ:!'

_output_shapes	
:ђ:%(!

_output_shapes
:	ђ: )
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
: ђ:!/

_output_shapes	
:ђ:.0*
(
_output_shapes
:ђђ:!1

_output_shapes	
:ђ:'2#
!
_output_shapes
:ђбђ:!3

_output_shapes	
:ђ:&4"
 
_output_shapes
:
ђђ:!5

_output_shapes	
:ђ:6

_output_shapes
: 
┬
`
D__inference_lambda_4_layer_call_and_return_conditional_losses_702497

inputs
identityЃ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЄ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1Є
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2§
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
■
«
$__inference_signature_wrapper_703250
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCallњ
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
GPU2 *0J 8ѓ **
f%R#
!__inference__wrapped_model_7019212
StatefulPartitionedCallј
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
┐
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_701987

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
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
AssignNewValue_1љ
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
Ж
Ё
-__inference_sequential_4_layer_call_fn_704324
lambda_4_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         ђ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_7025992
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

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
_user_specified_namelambda_4_input
г
h
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_702053

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
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
Ь
Л
6__inference_batch_normalization_4_layer_call_fn_704455

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCall║
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
GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7019432
StatefulPartitionedCallе
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
х
e
F__inference_dropout_14_layer_call_and_return_conditional_losses_702332

inputs
identityѕc
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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
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
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
В
Ё
-__inference_sequential_4_layer_call_fn_704225
lambda_4_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ
identityѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCalllambda_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:         ђ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_7022812
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

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
_user_specified_namelambda_4_input
Ў
Г
$__inference_CNN_layer_call_fn_703726

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCall»
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_7028632
StatefulPartitionedCallј
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
┐
│
E__inference_conv2d_12_layer_call_and_return_conditional_losses_702144

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
Relu¤
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulн
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp*
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
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ш
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_704583

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yК
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
┘
F
*__inference_flatten_4_layer_call_fn_704604

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђб* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_7022002
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         ђб2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
я
M
1__inference_max_pooling2d_13_layer_call_fn_702071

inputs
identityЫ
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
GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_7020652
PartitionedCallЈ
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
І
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_701943

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
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
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
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
Д
╝
!__inference__wrapped_model_701921
input_1

cnn_701887:

cnn_701889:

cnn_701891:

cnn_701893:$

cnn_701895: 

cnn_701897: %

cnn_701899: ђ

cnn_701901:	ђ&

cnn_701903:ђђ

cnn_701905:	ђ

cnn_701907:ђбђ

cnn_701909:	ђ

cnn_701911:
ђђ

cnn_701913:	ђ

cnn_701915:	ђ

cnn_701917:
identityѕбCNN/StatefulPartitionedCallЮ
CNN/StatefulPartitionedCallStatefulPartitionedCallinput_1
cnn_701887
cnn_701889
cnn_701891
cnn_701893
cnn_701895
cnn_701897
cnn_701899
cnn_701901
cnn_701903
cnn_701905
cnn_701907
cnn_701909
cnn_701911
cnn_701913
cnn_701915
cnn_701917*
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
GPU2 *0J 8ѓ * 
fR
__inference_call_6240122
CNN/StatefulPartitionedCallќ
IdentityIdentity$CNN/StatefulPartitionedCall:output:0^CNN/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
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
Зs
з
__inference_call_626070

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_12_conv2d_readvariableop_resource: D
6sequential_4_conv2d_12_biasadd_readvariableop_resource: P
5sequential_4_conv2d_13_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_13_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_14_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_14_biasadd_readvariableop_resource:	ђI
4sequential_4_dense_12_matmul_readvariableop_resource:ђбђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_12/BiasAdd/ReadVariableOpб,sequential_4/conv2d_12/Conv2D/ReadVariableOpб-sequential_4/conv2d_13/BiasAdd/ReadVariableOpб,sequential_4/conv2d_13/Conv2D/ReadVariableOpб-sequential_4/conv2d_14/BiasAdd/ReadVariableOpб,sequential_4/conv2d_14/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2с
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ђKK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1║
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:ђKK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_12/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_12/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK *
paddingSAME*
strides
2
sequential_4/conv2d_12/Conv2DЛ
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp▄
sequential_4/conv2d_12/BiasAddBiasAdd&sequential_4/conv2d_12/Conv2D:output:05sequential_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK 2 
sequential_4/conv2d_12/BiasAddЮ
sequential_4/conv2d_12/ReluRelu'sequential_4/conv2d_12/BiasAdd:output:0*
T0*'
_output_shapes
:ђKK 2
sequential_4/conv2d_12/Reluж
%sequential_4/max_pooling2d_12/MaxPoolMaxPool)sequential_4/conv2d_12/Relu:activations:0*'
_output_shapes
:ђ%% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_12/MaxPool█
,sequential_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_13/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_13/Conv2DConv2D.sequential_4/max_pooling2d_12/MaxPool:output:04sequential_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_13/Conv2Dм
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpП
sequential_4/conv2d_13/BiasAddBiasAdd&sequential_4/conv2d_13/Conv2D:output:05sequential_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ2 
sequential_4/conv2d_13/BiasAddъ
sequential_4/conv2d_13/ReluRelu'sequential_4/conv2d_13/BiasAdd:output:0*
T0*(
_output_shapes
:ђ%%ђ2
sequential_4/conv2d_13/ReluЖ
%sequential_4/max_pooling2d_13/MaxPoolMaxPool)sequential_4/conv2d_13/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_13/MaxPool▄
,sequential_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_14/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_14/Conv2DConv2D.sequential_4/max_pooling2d_13/MaxPool:output:04sequential_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_4/conv2d_14/Conv2Dм
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpП
sequential_4/conv2d_14/BiasAddBiasAdd&sequential_4/conv2d_14/Conv2D:output:05sequential_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_4/conv2d_14/BiasAddъ
sequential_4/conv2d_14/ReluRelu'sequential_4/conv2d_14/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_4/conv2d_14/ReluЖ
%sequential_4/max_pooling2d_14/MaxPoolMaxPool)sequential_4/conv2d_14/Relu:activations:0*(
_output_shapes
:ђ		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_14/MaxPool│
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_14/MaxPool:output:0*
T0*(
_output_shapes
:ђ		ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_4/flatten_4/Const╔
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*!
_output_shapes
:ђђб2 
sequential_4/flatten_4/Reshapeм
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp¤
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOpм
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/BiasAddЊ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/ReluЦ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOpЛ
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOpм
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/BiasAddЊ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/ReluЦ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOpЕ
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЮ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_14/BiasAddt
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*
_output_shapes
:	ђ2
dense_14/SoftmaxШ
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_12/BiasAdd/ReadVariableOp-^sequential_4/conv2d_12/Conv2D/ReadVariableOp.^sequential_4/conv2d_13/BiasAdd/ReadVariableOp-^sequential_4/conv2d_13/Conv2D/ReadVariableOp.^sequential_4/conv2d_14/BiasAdd/ReadVariableOp-^sequential_4/conv2d_14/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*
_output_shapes
:	ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ђKK: : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp-sequential_4/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_12/Conv2D/ReadVariableOp,sequential_4/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp-sequential_4/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_13/Conv2D/ReadVariableOp,sequential_4/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp-sequential_4/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_14/Conv2D/ReadVariableOp,sequential_4/conv2d_14/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:ђKK
 
_user_specified_nameinputs
Ѕv
ъ
H__inference_sequential_4_layer_call_and_return_conditional_losses_703901

inputs;
-batch_normalization_4_readvariableop_resource:=
/batch_normalization_4_readvariableop_1_resource:L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: C
(conv2d_13_conv2d_readvariableop_resource: ђ8
)conv2d_13_biasadd_readvariableop_resource:	ђD
(conv2d_14_conv2d_readvariableop_resource:ђђ8
)conv2d_14_biasadd_readvariableop_resource:	ђ<
'dense_12_matmul_readvariableop_resource:ђбђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identityѕб5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б conv2d_12/BiasAdd/ReadVariableOpбconv2d_12/Conv2D/ReadVariableOpб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб conv2d_13/BiasAdd/ReadVariableOpбconv2d_13/Conv2D/ReadVariableOpб conv2d_14/BiasAdd/ReadVariableOpбconv2d_14/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stackЎ
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1Ў
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2ф
lambda_4/strided_sliceStridedSliceinputs%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_4/strided_sliceХ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1у
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3│
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOpт
conv2d_12/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_12/Conv2Dф
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp░
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_12/Relu╩
max_pooling2d_12/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPool┤
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_13/Conv2D/ReadVariableOpП
conv2d_13/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_13/Conv2DФ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp▒
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_13/BiasAdd
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_13/Relu╦
max_pooling2d_13/MaxPoolMaxPoolconv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPoolх
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_14/Conv2D/ReadVariableOpП
conv2d_14/Conv2DConv2D!max_pooling2d_13/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_14/Conv2DФ
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp▒
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_14/Relu╦
max_pooling2d_14/MaxPoolMaxPoolconv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_14/MaxPoolћ
dropout_12/IdentityIdentity!max_pooling2d_14/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2
dropout_12/Identitys
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_4/ConstЮ
flatten_4/ReshapeReshapedropout_12/Identity:output:0flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2
flatten_4/ReshapeФ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/MatMulе
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_12/BiasAdd/ReadVariableOpд
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_12/Reluє
dropout_13/IdentityIdentitydense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_13/Identityф
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_13/MatMul/ReadVariableOpЦ
dense_13/MatMulMatMuldropout_13/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/MatMulе
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_13/BiasAdd/ReadVariableOpд
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_13/Reluє
dropout_14/IdentityIdentitydense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_14/Identity┘
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulЛ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulл
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulЦ
IdentityIdentitydropout_14/Identity:output:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
р
E
)__inference_lambda_4_layer_call_fn_704370

inputs
identity¤
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_7024972
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
я
M
1__inference_max_pooling2d_14_layer_call_fn_702083

inputs
identityЫ
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
GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_7020772
PartitionedCallЈ
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
І
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704388

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
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
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
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
Н
d
+__inference_dropout_13_layer_call_fn_704663

inputs
identityѕбStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_7023652
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
х
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_702365

inputs
identityѕc
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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
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
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
э
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_702470

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
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
┬
`
D__inference_lambda_4_layer_call_and_return_conditional_losses_704360

inputs
identityЃ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЄ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1Є
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2§
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
├
ю
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704424

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
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
epsilon%oЃ:*
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
г
h
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_702077

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
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
х
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_704653

inputs
identityѕc
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
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
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
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ќ
Г
$__inference_CNN_layer_call_fn_703763

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCallГ
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_7030032
StatefulPartitionedCallј
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
╣
г
D__inference_dense_13_layer_call_and_return_conditional_losses_704686

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
ReluК
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Сu
з
__inference_call_626142

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_12_conv2d_readvariableop_resource: D
6sequential_4_conv2d_12_biasadd_readvariableop_resource: P
5sequential_4_conv2d_13_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_13_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_14_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_14_biasadd_readvariableop_resource:	ђI
4sequential_4_dense_12_matmul_readvariableop_resource:ђбђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_12/BiasAdd/ReadVariableOpб,sequential_4/conv2d_12/Conv2D/ReadVariableOpб-sequential_4/conv2d_13/BiasAdd/ReadVariableOpб,sequential_4/conv2d_13/Conv2D/ReadVariableOpб-sequential_4/conv2d_14/BiasAdd/ReadVariableOpб,sequential_4/conv2d_14/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2в
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_12/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_12/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_12/Conv2DЛ
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpС
sequential_4/conv2d_12/BiasAddBiasAdd&sequential_4/conv2d_12/Conv2D:output:05sequential_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_12/BiasAddЦ
sequential_4/conv2d_12/ReluRelu'sequential_4/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_12/Reluы
%sequential_4/max_pooling2d_12/MaxPoolMaxPool)sequential_4/conv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_12/MaxPool█
,sequential_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_13/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_13/Conv2DConv2D.sequential_4/max_pooling2d_12/MaxPool:output:04sequential_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_13/Conv2Dм
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpт
sequential_4/conv2d_13/BiasAddBiasAdd&sequential_4/conv2d_13/Conv2D:output:05sequential_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_13/BiasAddд
sequential_4/conv2d_13/ReluRelu'sequential_4/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_13/ReluЫ
%sequential_4/max_pooling2d_13/MaxPoolMaxPool)sequential_4/conv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_13/MaxPool▄
,sequential_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_14/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_14/Conv2DConv2D.sequential_4/max_pooling2d_13/MaxPool:output:04sequential_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_14/Conv2Dм
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpт
sequential_4/conv2d_14/BiasAddBiasAdd&sequential_4/conv2d_14/Conv2D:output:05sequential_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_14/BiasAddд
sequential_4/conv2d_14/ReluRelu'sequential_4/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_14/ReluЫ
%sequential_4/max_pooling2d_14/MaxPoolMaxPool)sequential_4/conv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_14/MaxPool╗
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_14/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_4/flatten_4/ConstЛ
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_4/flatten_4/Reshapeм
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluГ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluГ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/Softmax■
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_12/BiasAdd/ReadVariableOp-^sequential_4/conv2d_12/Conv2D/ReadVariableOp.^sequential_4/conv2d_13/BiasAdd/ReadVariableOp-^sequential_4/conv2d_13/Conv2D/ReadVariableOp.^sequential_4/conv2d_14/BiasAdd/ReadVariableOp-^sequential_4/conv2d_14/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp-sequential_4/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_12/Conv2D/ReadVariableOp,sequential_4/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp-sequential_4/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_13/Conv2D/ReadVariableOp,sequential_4/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp-sequential_4/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_14/Conv2D/ReadVariableOp,sequential_4/conv2d_14/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
д
Л
6__inference_batch_normalization_4_layer_call_fn_704481

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallе
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
GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7021172
StatefulPartitionedCallќ
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
ж`
щ
H__inference_sequential_4_layer_call_and_return_conditional_losses_702599

inputs*
batch_normalization_4_702539:*
batch_normalization_4_702541:*
batch_normalization_4_702543:*
batch_normalization_4_702545:*
conv2d_12_702548: 
conv2d_12_702550: +
conv2d_13_702554: ђ
conv2d_13_702556:	ђ,
conv2d_14_702560:ђђ
conv2d_14_702562:	ђ$
dense_12_702568:ђбђ
dense_12_702570:	ђ#
dense_13_702574:
ђђ
dense_13_702576:	ђ
identityѕб-batch_normalization_4/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб!conv2d_13/StatefulPartitionedCallб!conv2d_14/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб1dense_12/kernel/Regularizer/Square/ReadVariableOpб dense_13/StatefulPartitionedCallб1dense_13/kernel/Regularizer/Square/ReadVariableOpб"dropout_12/StatefulPartitionedCallб"dropout_13/StatefulPartitionedCallб"dropout_14/StatefulPartitionedCallр
lambda_4/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_7024972
lambda_4/PartitionedCall╗
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0batch_normalization_4_702539batch_normalization_4_702541batch_normalization_4_702543batch_normalization_4_702545*
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
GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7024702/
-batch_normalization_4/StatefulPartitionedCallо
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_12_702548conv2d_12_702550*
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
GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_7021442#
!conv2d_12/StatefulPartitionedCallЮ
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_7020532"
 max_pooling2d_12/PartitionedCall╩
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_702554conv2d_13_702556*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_7021622#
!conv2d_13/StatefulPartitionedCallъ
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_7020652"
 max_pooling2d_13/PartitionedCall╩
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_14_702560conv2d_14_702562*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_7021802#
!conv2d_14/StatefulPartitionedCallъ
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_7020772"
 max_pooling2d_14/PartitionedCallБ
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_7024042$
"dropout_12/StatefulPartitionedCallЃ
flatten_4/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђб* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_7022002
flatten_4/PartitionedCallХ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_702568dense_12_702570*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_7022192"
 dense_12/StatefulPartitionedCall└
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_7023652$
"dropout_13/StatefulPartitionedCall┐
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_13_702574dense_13_702576*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_7022492"
 dense_13/StatefulPartitionedCall└
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0#^dropout_13/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_7023322$
"dropout_14/StatefulPartitionedCall┴
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_702548*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul╣
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_702568*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulИ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_702574* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulЬ
IdentityIdentity+dropout_14/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
У
│
__inference_loss_fn_2_704755N
:dense_13_kernel_regularizer_square_readvariableop_resource:
ђђ
identityѕб1dense_13/kernel/Regularizer/Square/ReadVariableOpс
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_13_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulџ
IdentityIdentity#dense_13/kernel/Regularizer/mul:z:02^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp
Н
d
+__inference_dropout_14_layer_call_fn_704722

inputs
identityѕбStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_7023322
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ќ
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_704571

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         		ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
╔
G
+__inference_dropout_14_layer_call_fn_704717

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
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_7022602
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
в
┤
__inference_loss_fn_1_704744O
:dense_12_kernel_regularizer_square_readvariableop_resource:ђбђ
identityѕб1dense_12/kernel/Regularizer/Square/ReadVariableOpС
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_12_kernel_regularizer_square_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulџ
IdentityIdentity#dense_12/kernel/Regularizer/mul:z:02^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp
э
d
F__inference_dropout_14_layer_call_and_return_conditional_losses_704700

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╔
G
+__inference_dropout_13_layer_call_fn_704658

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
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_7022302
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╩
Ъ
*__inference_conv2d_12_layer_call_fn_704526

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallѓ
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
GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_7021442
StatefulPartitionedCallќ
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
┬
`
D__inference_lambda_4_layer_call_and_return_conditional_losses_702098

inputs
identityЃ
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stackЄ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1Є
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2§
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
И

Ш
D__inference_dense_14_layer_call_and_return_conditional_losses_704335

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Сu
з
__inference_call_624012

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_12_conv2d_readvariableop_resource: D
6sequential_4_conv2d_12_biasadd_readvariableop_resource: P
5sequential_4_conv2d_13_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_13_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_14_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_14_biasadd_readvariableop_resource:	ђI
4sequential_4_dense_12_matmul_readvariableop_resource:ђбђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_12/BiasAdd/ReadVariableOpб,sequential_4/conv2d_12/Conv2D/ReadVariableOpб-sequential_4/conv2d_13/BiasAdd/ReadVariableOpб,sequential_4/conv2d_13/Conv2D/ReadVariableOpб-sequential_4/conv2d_14/BiasAdd/ReadVariableOpб,sequential_4/conv2d_14/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2в
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_12/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_12/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_12/Conv2DЛ
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpС
sequential_4/conv2d_12/BiasAddBiasAdd&sequential_4/conv2d_12/Conv2D:output:05sequential_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_12/BiasAddЦ
sequential_4/conv2d_12/ReluRelu'sequential_4/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_12/Reluы
%sequential_4/max_pooling2d_12/MaxPoolMaxPool)sequential_4/conv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_12/MaxPool█
,sequential_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_13/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_13/Conv2DConv2D.sequential_4/max_pooling2d_12/MaxPool:output:04sequential_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_13/Conv2Dм
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpт
sequential_4/conv2d_13/BiasAddBiasAdd&sequential_4/conv2d_13/Conv2D:output:05sequential_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_13/BiasAddд
sequential_4/conv2d_13/ReluRelu'sequential_4/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_13/ReluЫ
%sequential_4/max_pooling2d_13/MaxPoolMaxPool)sequential_4/conv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_13/MaxPool▄
,sequential_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_14/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_14/Conv2DConv2D.sequential_4/max_pooling2d_13/MaxPool:output:04sequential_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_14/Conv2Dм
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpт
sequential_4/conv2d_14/BiasAddBiasAdd&sequential_4/conv2d_14/Conv2D:output:05sequential_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_14/BiasAddд
sequential_4/conv2d_14/ReluRelu'sequential_4/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_14/ReluЫ
%sequential_4/max_pooling2d_14/MaxPoolMaxPool)sequential_4/conv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_14/MaxPool╗
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_14/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_4/flatten_4/ConstЛ
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_4/flatten_4/Reshapeм
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluГ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluГ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/Softmax■
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_12/BiasAdd/ReadVariableOp-^sequential_4/conv2d_12/Conv2D/ReadVariableOp.^sequential_4/conv2d_13/BiasAdd/ReadVariableOp-^sequential_4/conv2d_13/Conv2D/ReadVariableOp.^sequential_4/conv2d_14/BiasAdd/ReadVariableOp-^sequential_4/conv2d_14/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp-sequential_4/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_12/Conv2D/ReadVariableOp,sequential_4/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp-sequential_4/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_13/Conv2D/ReadVariableOp,sequential_4/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp-sequential_4/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_14/Conv2D/ReadVariableOp,sequential_4/conv2d_14/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ќ
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_702192

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         		ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         		ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
┐
Г
D__inference_dense_12_layer_call_and_return_conditional_losses_702219

inputs3
matmul_readvariableop_resource:ђбђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpљ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Relu╚
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђб: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:         ђб
 
_user_specified_nameinputs
Аv
д
H__inference_sequential_4_layer_call_and_return_conditional_losses_704088
lambda_4_input;
-batch_normalization_4_readvariableop_resource:=
/batch_normalization_4_readvariableop_1_resource:L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: C
(conv2d_13_conv2d_readvariableop_resource: ђ8
)conv2d_13_biasadd_readvariableop_resource:	ђD
(conv2d_14_conv2d_readvariableop_resource:ђђ8
)conv2d_14_biasadd_readvariableop_resource:	ђ<
'dense_12_matmul_readvariableop_resource:ђбђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identityѕб5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б conv2d_12/BiasAdd/ReadVariableOpбconv2d_12/Conv2D/ReadVariableOpб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб conv2d_13/BiasAdd/ReadVariableOpбconv2d_13/Conv2D/ReadVariableOpб conv2d_14/BiasAdd/ReadVariableOpбconv2d_14/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stackЎ
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1Ў
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2▓
lambda_4/strided_sliceStridedSlicelambda_4_input%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_4/strided_sliceХ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1у
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3│
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOpт
conv2d_12/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_12/Conv2Dф
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp░
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_12/Relu╩
max_pooling2d_12/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPool┤
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_13/Conv2D/ReadVariableOpП
conv2d_13/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_13/Conv2DФ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp▒
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_13/BiasAdd
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_13/Relu╦
max_pooling2d_13/MaxPoolMaxPoolconv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPoolх
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_14/Conv2D/ReadVariableOpП
conv2d_14/Conv2DConv2D!max_pooling2d_13/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_14/Conv2DФ
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp▒
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_14/Relu╦
max_pooling2d_14/MaxPoolMaxPoolconv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_14/MaxPoolћ
dropout_12/IdentityIdentity!max_pooling2d_14/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2
dropout_12/Identitys
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_4/ConstЮ
flatten_4/ReshapeReshapedropout_12/Identity:output:0flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2
flatten_4/ReshapeФ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/MatMulе
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_12/BiasAdd/ReadVariableOpд
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_12/Reluє
dropout_13/IdentityIdentitydense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_13/Identityф
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_13/MatMul/ReadVariableOpЦ
dense_13/MatMulMatMuldropout_13/Identity:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/MatMulе
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_13/BiasAdd/ReadVariableOpд
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_13/Reluє
dropout_14/IdentityIdentitydense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_14/Identity┘
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulЛ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulл
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulЦ
IdentityIdentitydropout_14/Identity:output:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:_ [
/
_output_shapes
:         KK
(
_user_specified_namelambda_4_input
╣џ
Ь
H__inference_sequential_4_layer_call_and_return_conditional_losses_704005

inputs;
-batch_normalization_4_readvariableop_resource:=
/batch_normalization_4_readvariableop_1_resource:L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_12_conv2d_readvariableop_resource: 7
)conv2d_12_biasadd_readvariableop_resource: C
(conv2d_13_conv2d_readvariableop_resource: ђ8
)conv2d_13_biasadd_readvariableop_resource:	ђD
(conv2d_14_conv2d_readvariableop_resource:ђђ8
)conv2d_14_biasadd_readvariableop_resource:	ђ<
'dense_12_matmul_readvariableop_resource:ђбђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identityѕб$batch_normalization_4/AssignNewValueб&batch_normalization_4/AssignNewValue_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б conv2d_12/BiasAdd/ReadVariableOpбconv2d_12/Conv2D/ReadVariableOpб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб conv2d_13/BiasAdd/ReadVariableOpбconv2d_13/Conv2D/ReadVariableOpб conv2d_14/BiasAdd/ReadVariableOpбconv2d_14/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЋ
lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
lambda_4/strided_slice/stackЎ
lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2 
lambda_4/strided_slice/stack_1Ў
lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2 
lambda_4/strided_slice/stack_2ф
lambda_4/strided_sliceStridedSliceinputs%lambda_4/strided_slice/stack:output:0'lambda_4/strided_slice/stack_1:output:0'lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2
lambda_4/strided_sliceХ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_4/ReadVariableOp╝
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3lambda_4/strided_slice:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_4/FusedBatchNormV3░
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue╝
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1│
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_12/Conv2D/ReadVariableOpт
conv2d_12/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
conv2d_12/Conv2Dф
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp░
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
conv2d_12/Relu╩
max_pooling2d_12/MaxPoolMaxPoolconv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPool┤
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02!
conv2d_13/Conv2D/ReadVariableOpП
conv2d_13/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
conv2d_13/Conv2DФ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp▒
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_13/BiasAdd
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
conv2d_13/Relu╦
max_pooling2d_13/MaxPoolMaxPoolconv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPoolх
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_14/Conv2D/ReadVariableOpП
conv2d_14/Conv2DConv2D!max_pooling2d_13/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_14/Conv2DФ
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp▒
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_14/BiasAdd
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_14/Relu╦
max_pooling2d_14/MaxPoolMaxPoolconv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_14/MaxPooly
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout_12/dropout/ConstИ
dropout_12/dropout/MulMul!max_pooling2d_14/MaxPool:output:0!dropout_12/dropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2
dropout_12/dropout/MulЁ
dropout_12/dropout/ShapeShape!max_pooling2d_14/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shapeя
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype021
/dropout_12/dropout/random_uniform/RandomUniformІ
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2#
!dropout_12/dropout/GreaterEqual/yз
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2!
dropout_12/dropout/GreaterEqualЕ
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2
dropout_12/dropout/Cast»
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2
dropout_12/dropout/Mul_1s
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
flatten_4/ConstЮ
flatten_4/ReshapeReshapedropout_12/dropout/Mul_1:z:0flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2
flatten_4/ReshapeФ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02 
dense_12/MatMul/ReadVariableOpБ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/MatMulе
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_12/BiasAdd/ReadVariableOpд
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_12/BiasAddt
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_12/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_13/dropout/Constф
dropout_13/dropout/MulMuldense_12/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_13/dropout/Mul
dropout_13/dropout/ShapeShapedense_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shapeо
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_13/dropout/random_uniform/RandomUniformІ
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_13/dropout/GreaterEqual/yв
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_13/dropout/GreaterEqualА
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_13/dropout/CastД
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_13/dropout/Mul_1ф
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02 
dense_13/MatMul/ReadVariableOpЦ
dense_13/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/MatMulе
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_13/BiasAdd/ReadVariableOpд
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_13/Reluy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_14/dropout/Constф
dropout_14/dropout/MulMuldense_13/Relu:activations:0!dropout_14/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_14/dropout/Mul
dropout_14/dropout/ShapeShapedense_13/Relu:activations:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shapeо
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype021
/dropout_14/dropout/random_uniform/RandomUniformІ
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_14/dropout/GreaterEqual/yв
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2!
dropout_14/dropout/GreaterEqualА
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_14/dropout/CastД
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_14/dropout/Mul_1┘
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulЛ
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulл
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulш
IdentityIdentitydropout_14/dropout/Mul_1:z:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
р
E
)__inference_lambda_4_layer_call_fn_704365

inputs
identity¤
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_7020982
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
щ┴
Б
?__inference_CNN_layer_call_and_return_conditional_losses_703652
input_1H
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_12_conv2d_readvariableop_resource: D
6sequential_4_conv2d_12_biasadd_readvariableop_resource: P
5sequential_4_conv2d_13_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_13_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_14_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_14_biasadd_readvariableop_resource:	ђI
4sequential_4_dense_12_matmul_readvariableop_resource:ђбђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpб1sequential_4/batch_normalization_4/AssignNewValueб3sequential_4/batch_normalization_4/AssignNewValue_1бBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_12/BiasAdd/ReadVariableOpб,sequential_4/conv2d_12/Conv2D/ReadVariableOpб-sequential_4/conv2d_13/BiasAdd/ReadVariableOpб,sequential_4/conv2d_13/Conv2D/ReadVariableOpб-sequential_4/conv2d_14/BiasAdd/ReadVariableOpб,sequential_4/conv2d_14/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2В
#sequential_4/lambda_4/strided_sliceStridedSliceinput_12sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1л
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<25
3sequential_4/batch_normalization_4/FusedBatchNormV3ы
1sequential_4/batch_normalization_4/AssignNewValueAssignVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource@sequential_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0C^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype023
1sequential_4/batch_normalization_4/AssignNewValue§
3sequential_4/batch_normalization_4/AssignNewValue_1AssignVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceDsequential_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0E^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype025
3sequential_4/batch_normalization_4/AssignNewValue_1┌
,sequential_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_12/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_12/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_12/Conv2DЛ
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpС
sequential_4/conv2d_12/BiasAddBiasAdd&sequential_4/conv2d_12/Conv2D:output:05sequential_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_12/BiasAddЦ
sequential_4/conv2d_12/ReluRelu'sequential_4/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_12/Reluы
%sequential_4/max_pooling2d_12/MaxPoolMaxPool)sequential_4/conv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_12/MaxPool█
,sequential_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_13/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_13/Conv2DConv2D.sequential_4/max_pooling2d_12/MaxPool:output:04sequential_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_13/Conv2Dм
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpт
sequential_4/conv2d_13/BiasAddBiasAdd&sequential_4/conv2d_13/Conv2D:output:05sequential_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_13/BiasAddд
sequential_4/conv2d_13/ReluRelu'sequential_4/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_13/ReluЫ
%sequential_4/max_pooling2d_13/MaxPoolMaxPool)sequential_4/conv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_13/MaxPool▄
,sequential_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_14/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_14/Conv2DConv2D.sequential_4/max_pooling2d_13/MaxPool:output:04sequential_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_14/Conv2Dм
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpт
sequential_4/conv2d_14/BiasAddBiasAdd&sequential_4/conv2d_14/Conv2D:output:05sequential_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_14/BiasAddд
sequential_4/conv2d_14/ReluRelu'sequential_4/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_14/ReluЫ
%sequential_4/max_pooling2d_14/MaxPoolMaxPool)sequential_4/conv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_14/MaxPoolЊ
%sequential_4/dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2'
%sequential_4/dropout_12/dropout/ConstВ
#sequential_4/dropout_12/dropout/MulMul.sequential_4/max_pooling2d_14/MaxPool:output:0.sequential_4/dropout_12/dropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2%
#sequential_4/dropout_12/dropout/Mulг
%sequential_4/dropout_12/dropout/ShapeShape.sequential_4/max_pooling2d_14/MaxPool:output:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_12/dropout/ShapeЁ
<sequential_4/dropout_12/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_12/dropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype02>
<sequential_4/dropout_12/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=20
.sequential_4/dropout_12/dropout/GreaterEqual/yД
,sequential_4/dropout_12/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_12/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_12/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2.
,sequential_4/dropout_12/dropout/GreaterEqualл
$sequential_4/dropout_12/dropout/CastCast0sequential_4/dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2&
$sequential_4/dropout_12/dropout/Castс
%sequential_4/dropout_12/dropout/Mul_1Mul'sequential_4/dropout_12/dropout/Mul:z:0(sequential_4/dropout_12/dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2'
%sequential_4/dropout_12/dropout/Mul_1Ї
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_4/flatten_4/ConstЛ
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/dropout/Mul_1:z:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_4/flatten_4/Reshapeм
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluЊ
%sequential_4/dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_4/dropout_13/dropout/Constя
#sequential_4/dropout_13/dropout/MulMul(sequential_4/dense_12/Relu:activations:0.sequential_4/dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_4/dropout_13/dropout/Mulд
%sequential_4/dropout_13/dropout/ShapeShape(sequential_4/dense_12/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_13/dropout/Shape§
<sequential_4/dropout_13/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_13/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_4/dropout_13/dropout/GreaterEqual/yЪ
,sequential_4/dropout_13/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_13/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_4/dropout_13/dropout/GreaterEqual╚
$sequential_4/dropout_13/dropout/CastCast0sequential_4/dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_4/dropout_13/dropout/Cast█
%sequential_4/dropout_13/dropout/Mul_1Mul'sequential_4/dropout_13/dropout/Mul:z:0(sequential_4/dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_4/dropout_13/dropout/Mul_1Л
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/dropout/Mul_1:z:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluЊ
%sequential_4/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%sequential_4/dropout_14/dropout/Constя
#sequential_4/dropout_14/dropout/MulMul(sequential_4/dense_13/Relu:activations:0.sequential_4/dropout_14/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2%
#sequential_4/dropout_14/dropout/Mulд
%sequential_4/dropout_14/dropout/ShapeShape(sequential_4/dense_13/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_4/dropout_14/dropout/Shape§
<sequential_4/dropout_14/dropout/random_uniform/RandomUniformRandomUniform.sequential_4/dropout_14/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02>
<sequential_4/dropout_14/dropout/random_uniform/RandomUniformЦ
.sequential_4/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_4/dropout_14/dropout/GreaterEqual/yЪ
,sequential_4/dropout_14/dropout/GreaterEqualGreaterEqualEsequential_4/dropout_14/dropout/random_uniform/RandomUniform:output:07sequential_4/dropout_14/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2.
,sequential_4/dropout_14/dropout/GreaterEqual╚
$sequential_4/dropout_14/dropout/CastCast0sequential_4/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2&
$sequential_4/dropout_14/dropout/Cast█
%sequential_4/dropout_14/dropout/Mul_1Mul'sequential_4/dropout_14/dropout/Mul:z:0(sequential_4/dropout_14/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2'
%sequential_4/dropout_14/dropout/Mul_1Е
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/dropout/Mul_1:z:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/SoftmaxТ
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulя
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulП
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulЁ	
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_12/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp2^sequential_4/batch_normalization_4/AssignNewValue4^sequential_4/batch_normalization_4/AssignNewValue_1C^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_12/BiasAdd/ReadVariableOp-^sequential_4/conv2d_12/Conv2D/ReadVariableOp.^sequential_4/conv2d_13/BiasAdd/ReadVariableOp-^sequential_4/conv2d_13/Conv2D/ReadVariableOp.^sequential_4/conv2d_14/BiasAdd/ReadVariableOp-^sequential_4/conv2d_14/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2f
1sequential_4/batch_normalization_4/AssignNewValue1sequential_4/batch_normalization_4/AssignNewValue2j
3sequential_4/batch_normalization_4/AssignNewValue_13sequential_4/batch_normalization_4/AssignNewValue_12ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp-sequential_4/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_12/Conv2D/ReadVariableOp,sequential_4/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp-sequential_4/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_13/Conv2D/ReadVariableOp,sequential_4/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp-sequential_4/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_14/Conv2D/ReadVariableOp,sequential_4/conv2d_14/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:X T
/
_output_shapes
:         KK
!
_user_specified_name	input_1
џ
«
$__inference_CNN_layer_call_fn_703800
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCall«
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_7030032
StatefulPartitionedCallј
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
Зs
з
__inference_call_625998

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_12_conv2d_readvariableop_resource: D
6sequential_4_conv2d_12_biasadd_readvariableop_resource: P
5sequential_4_conv2d_13_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_13_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_14_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_14_biasadd_readvariableop_resource:	ђI
4sequential_4_dense_12_matmul_readvariableop_resource:ђбђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_12/BiasAdd/ReadVariableOpб,sequential_4/conv2d_12/Conv2D/ReadVariableOpб-sequential_4/conv2d_13/BiasAdd/ReadVariableOpб,sequential_4/conv2d_13/Conv2D/ReadVariableOpб-sequential_4/conv2d_14/BiasAdd/ReadVariableOpб,sequential_4/conv2d_14/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2с
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ђKK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1║
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*C
_output_shapes1
/:ђKK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_12/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_12/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK *
paddingSAME*
strides
2
sequential_4/conv2d_12/Conv2DЛ
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp▄
sequential_4/conv2d_12/BiasAddBiasAdd&sequential_4/conv2d_12/Conv2D:output:05sequential_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ђKK 2 
sequential_4/conv2d_12/BiasAddЮ
sequential_4/conv2d_12/ReluRelu'sequential_4/conv2d_12/BiasAdd:output:0*
T0*'
_output_shapes
:ђKK 2
sequential_4/conv2d_12/Reluж
%sequential_4/max_pooling2d_12/MaxPoolMaxPool)sequential_4/conv2d_12/Relu:activations:0*'
_output_shapes
:ђ%% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_12/MaxPool█
,sequential_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_13/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_13/Conv2DConv2D.sequential_4/max_pooling2d_12/MaxPool:output:04sequential_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_13/Conv2Dм
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpП
sequential_4/conv2d_13/BiasAddBiasAdd&sequential_4/conv2d_13/Conv2D:output:05sequential_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђ%%ђ2 
sequential_4/conv2d_13/BiasAddъ
sequential_4/conv2d_13/ReluRelu'sequential_4/conv2d_13/BiasAdd:output:0*
T0*(
_output_shapes
:ђ%%ђ2
sequential_4/conv2d_13/ReluЖ
%sequential_4/max_pooling2d_13/MaxPoolMaxPool)sequential_4/conv2d_13/Relu:activations:0*(
_output_shapes
:ђђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_13/MaxPool▄
,sequential_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_14/Conv2D/ReadVariableOpЅ
sequential_4/conv2d_14/Conv2DConv2D.sequential_4/max_pooling2d_13/MaxPool:output:04sequential_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ*
paddingSAME*
strides
2
sequential_4/conv2d_14/Conv2Dм
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpП
sequential_4/conv2d_14/BiasAddBiasAdd&sequential_4/conv2d_14/Conv2D:output:05sequential_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ђђ2 
sequential_4/conv2d_14/BiasAddъ
sequential_4/conv2d_14/ReluRelu'sequential_4/conv2d_14/BiasAdd:output:0*
T0*(
_output_shapes
:ђђ2
sequential_4/conv2d_14/ReluЖ
%sequential_4/max_pooling2d_14/MaxPoolMaxPool)sequential_4/conv2d_14/Relu:activations:0*(
_output_shapes
:ђ		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_14/MaxPool│
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_14/MaxPool:output:0*
T0*(
_output_shapes
:ђ		ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_4/flatten_4/Const╔
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*!
_output_shapes
:ђђб2 
sequential_4/flatten_4/Reshapeм
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOp¤
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOpм
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/BiasAddЊ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_12/ReluЦ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOpЛ
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOpм
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/BiasAddЊ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0* 
_output_shapes
:
ђђ2
sequential_4/dense_13/ReluЦ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0* 
_output_shapes
:
ђђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOpЕ
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЮ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ2
dense_14/BiasAddt
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*
_output_shapes
:	ђ2
dense_14/SoftmaxШ
IdentityIdentitydense_14/Softmax:softmax:0 ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_12/BiasAdd/ReadVariableOp-^sequential_4/conv2d_12/Conv2D/ReadVariableOp.^sequential_4/conv2d_13/BiasAdd/ReadVariableOp-^sequential_4/conv2d_13/Conv2D/ReadVariableOp.^sequential_4/conv2d_14/BiasAdd/ReadVariableOp-^sequential_4/conv2d_14/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*
_output_shapes
:	ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ђKK: : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp-sequential_4/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_12/Conv2D/ReadVariableOp,sequential_4/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp-sequential_4/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_13/Conv2D/ReadVariableOp,sequential_4/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp-sequential_4/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_14/Conv2D/ReadVariableOp,sequential_4/conv2d_14/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:O K
'
_output_shapes
:ђKK
 
_user_specified_nameinputs
ш
d
+__inference_dropout_12_layer_call_fn_704593

inputs
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_7024042
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
ю
«
$__inference_CNN_layer_call_fn_703689
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:
identityѕбStatefulPartitionedCall░
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
GPU2 *0J 8ѓ *H
fCRA
?__inference_CNN_layer_call_and_return_conditional_losses_7028632
StatefulPartitionedCallј
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
Д
Ў
)__inference_dense_13_layer_call_fn_704695

inputs
unknown:
ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_7022492
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
э
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_702230

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
─▀
╠!
"__inference__traced_restore_705106
file_prefix3
 assignvariableop_dense_14_kernel:	ђ.
 assignvariableop_1_dense_14_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: <
.assignvariableop_7_batch_normalization_4_gamma:;
-assignvariableop_8_batch_normalization_4_beta:B
4assignvariableop_9_batch_normalization_4_moving_mean:G
9assignvariableop_10_batch_normalization_4_moving_variance:>
$assignvariableop_11_conv2d_12_kernel: 0
"assignvariableop_12_conv2d_12_bias: ?
$assignvariableop_13_conv2d_13_kernel: ђ1
"assignvariableop_14_conv2d_13_bias:	ђ@
$assignvariableop_15_conv2d_14_kernel:ђђ1
"assignvariableop_16_conv2d_14_bias:	ђ8
#assignvariableop_17_dense_12_kernel:ђбђ0
!assignvariableop_18_dense_12_bias:	ђ7
#assignvariableop_19_dense_13_kernel:
ђђ0
!assignvariableop_20_dense_13_bias:	ђ#
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: =
*assignvariableop_25_adam_dense_14_kernel_m:	ђ6
(assignvariableop_26_adam_dense_14_bias_m:D
6assignvariableop_27_adam_batch_normalization_4_gamma_m:C
5assignvariableop_28_adam_batch_normalization_4_beta_m:E
+assignvariableop_29_adam_conv2d_12_kernel_m: 7
)assignvariableop_30_adam_conv2d_12_bias_m: F
+assignvariableop_31_adam_conv2d_13_kernel_m: ђ8
)assignvariableop_32_adam_conv2d_13_bias_m:	ђG
+assignvariableop_33_adam_conv2d_14_kernel_m:ђђ8
)assignvariableop_34_adam_conv2d_14_bias_m:	ђ?
*assignvariableop_35_adam_dense_12_kernel_m:ђбђ7
(assignvariableop_36_adam_dense_12_bias_m:	ђ>
*assignvariableop_37_adam_dense_13_kernel_m:
ђђ7
(assignvariableop_38_adam_dense_13_bias_m:	ђ=
*assignvariableop_39_adam_dense_14_kernel_v:	ђ6
(assignvariableop_40_adam_dense_14_bias_v:D
6assignvariableop_41_adam_batch_normalization_4_gamma_v:C
5assignvariableop_42_adam_batch_normalization_4_beta_v:E
+assignvariableop_43_adam_conv2d_12_kernel_v: 7
)assignvariableop_44_adam_conv2d_12_bias_v: F
+assignvariableop_45_adam_conv2d_13_kernel_v: ђ8
)assignvariableop_46_adam_conv2d_13_bias_v:	ђG
+assignvariableop_47_adam_conv2d_14_kernel_v:ђђ8
)assignvariableop_48_adam_conv2d_14_bias_v:	ђ?
*assignvariableop_49_adam_dense_12_kernel_v:ђбђ7
(assignvariableop_50_adam_dense_12_bias_v:	ђ>
*assignvariableop_51_adam_dense_13_kernel_v:
ђђ7
(assignvariableop_52_adam_dense_13_bias_v:	ђ
identity_54ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9В
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*Э
valueЬBв6B)_output/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_output/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBE_output/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_output/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЩ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╝
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ь
_output_shapes█
п::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЪ
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2А
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Б
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Б
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5б
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ф
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_4_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8▓
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_4_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╣
AssignVariableOp_9AssignVariableOp4assignvariableop_9_batch_normalization_4_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┴
AssignVariableOp_10AssignVariableOp9assignvariableop_10_batch_normalization_4_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11г
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_12_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ф
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_12_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13г
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_13_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ф
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_13_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15г
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_14_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ф
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_14_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ф
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_12_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Е
AssignVariableOp_18AssignVariableOp!assignvariableop_18_dense_12_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ф
AssignVariableOp_19AssignVariableOp#assignvariableop_19_dense_13_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Е
AssignVariableOp_20AssignVariableOp!assignvariableop_20_dense_13_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21А
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Б
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Б
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
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_14_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_14_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Й
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_batch_normalization_4_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28й
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_batch_normalization_4_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29│
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_12_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▒
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_12_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31│
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_13_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_13_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_14_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_14_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▓
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_12_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36░
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_12_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_13_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38░
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_13_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39▓
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_14_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40░
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_14_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Й
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_4_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42й
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_4_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43│
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_12_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44▒
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_12_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45│
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_13_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_13_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_14_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_14_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49▓
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_12_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50░
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_12_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51▓
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_13_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52░
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_13_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_529
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpВ	
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
э
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704442

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1п
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
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
г
h
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_702065

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
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
э
d
F__inference_dropout_14_layer_call_and_return_conditional_losses_702260

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╣
г
D__inference_dense_13_layer_call_and_return_conditional_losses_702249

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
ReluК
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mul╠
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
А
Ђ
E__inference_conv2d_14_layer_call_and_return_conditional_losses_704557

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
В
Л
6__inference_batch_normalization_4_layer_call_fn_704468

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallИ
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
GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7019872
StatefulPartitionedCallе
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
ц
Л
6__inference_batch_normalization_4_layer_call_fn_704494

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallд
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
GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7024702
StatefulPartitionedCallќ
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
ж
G
+__inference_dropout_12_layer_call_fn_704588

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_7021922
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
я
M
1__inference_max_pooling2d_12_layer_call_fn_702059

inputs
identityЫ
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
GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_7020532
PartitionedCallЈ
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
в
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_704599

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђб2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђб2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs
м
§
-__inference_sequential_4_layer_call_fn_704291

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: $
	unknown_5: ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:ђбђ

unknown_10:	ђ

unknown_11:
ђђ

unknown_12:	ђ
identityѕбStatefulPartitionedCallЏ
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
:         ђ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_7025992
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

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
┐
└
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704406

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
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
AssignNewValue_1љ
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
Ё\
і
H__inference_sequential_4_layer_call_and_return_conditional_losses_702281

inputs*
batch_normalization_4_702118:*
batch_normalization_4_702120:*
batch_normalization_4_702122:*
batch_normalization_4_702124:*
conv2d_12_702145: 
conv2d_12_702147: +
conv2d_13_702163: ђ
conv2d_13_702165:	ђ,
conv2d_14_702181:ђђ
conv2d_14_702183:	ђ$
dense_12_702220:ђбђ
dense_12_702222:	ђ#
dense_13_702250:
ђђ
dense_13_702252:	ђ
identityѕб-batch_normalization_4/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб!conv2d_13/StatefulPartitionedCallб!conv2d_14/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб1dense_12/kernel/Regularizer/Square/ReadVariableOpб dense_13/StatefulPartitionedCallб1dense_13/kernel/Regularizer/Square/ReadVariableOpр
lambda_4/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_lambda_4_layer_call_and_return_conditional_losses_7020982
lambda_4/PartitionedCallй
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall!lambda_4/PartitionedCall:output:0batch_normalization_4_702118batch_normalization_4_702120batch_normalization_4_702122batch_normalization_4_702124*
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
GPU2 *0J 8ѓ *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_7021172/
-batch_normalization_4/StatefulPartitionedCallо
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_12_702145conv2d_12_702147*
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
GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_7021442#
!conv2d_12/StatefulPartitionedCallЮ
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
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
GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_7020532"
 max_pooling2d_12/PartitionedCall╩
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_702163conv2d_13_702165*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_7021622#
!conv2d_13/StatefulPartitionedCallъ
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_7020652"
 max_pooling2d_13/PartitionedCall╩
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_14_702181conv2d_14_702183*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_7021802#
!conv2d_14/StatefulPartitionedCallъ
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_7020772"
 max_pooling2d_14/PartitionedCallІ
dropout_12/PartitionedCallPartitionedCall)max_pooling2d_14/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_7021922
dropout_12/PartitionedCallч
flatten_4/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђб* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_7022002
flatten_4/PartitionedCallХ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_702220dense_12_702222*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_7022192"
 dense_12/StatefulPartitionedCallЃ
dropout_13/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_7022302
dropout_13/PartitionedCallи
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_13_702250dense_13_702252*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_7022492"
 dense_13/StatefulPartitionedCallЃ
dropout_14/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_dropout_14_layer_call_and_return_conditional_losses_7022602
dropout_14/PartitionedCall┴
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_702145*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mul╣
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_702220*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulИ
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_702250* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulэ
IdentityIdentity#dropout_14/PartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/Square/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         KK: : : : : : : : : : : : : : 2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
А
Ђ
E__inference_conv2d_14_layer_call_and_return_conditional_losses_702180

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ю
ђ
E__inference_conv2d_13_layer_call_and_return_conditional_losses_702162

inputs9
conv2d_readvariableop_resource: ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
Reluа
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         %%ђ2

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
§ћ
И
?__inference_CNN_layer_call_and_return_conditional_losses_703340

inputsH
:sequential_4_batch_normalization_4_readvariableop_resource:J
<sequential_4_batch_normalization_4_readvariableop_1_resource:Y
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:[
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_12_conv2d_readvariableop_resource: D
6sequential_4_conv2d_12_biasadd_readvariableop_resource: P
5sequential_4_conv2d_13_conv2d_readvariableop_resource: ђE
6sequential_4_conv2d_13_biasadd_readvariableop_resource:	ђQ
5sequential_4_conv2d_14_conv2d_readvariableop_resource:ђђE
6sequential_4_conv2d_14_biasadd_readvariableop_resource:	ђI
4sequential_4_dense_12_matmul_readvariableop_resource:ђбђD
5sequential_4_dense_12_biasadd_readvariableop_resource:	ђH
4sequential_4_dense_13_matmul_readvariableop_resource:
ђђD
5sequential_4_dense_13_biasadd_readvariableop_resource:	ђ:
'dense_14_matmul_readvariableop_resource:	ђ6
(dense_14_biasadd_readvariableop_resource:
identityѕб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpбdense_14/BiasAdd/ReadVariableOpбdense_14/MatMul/ReadVariableOpбBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_4/batch_normalization_4/ReadVariableOpб3sequential_4/batch_normalization_4/ReadVariableOp_1б-sequential_4/conv2d_12/BiasAdd/ReadVariableOpб,sequential_4/conv2d_12/Conv2D/ReadVariableOpб-sequential_4/conv2d_13/BiasAdd/ReadVariableOpб,sequential_4/conv2d_13/Conv2D/ReadVariableOpб-sequential_4/conv2d_14/BiasAdd/ReadVariableOpб,sequential_4/conv2d_14/Conv2D/ReadVariableOpб,sequential_4/dense_12/BiasAdd/ReadVariableOpб+sequential_4/dense_12/MatMul/ReadVariableOpб,sequential_4/dense_13/BiasAdd/ReadVariableOpб+sequential_4/dense_13/MatMul/ReadVariableOp»
)sequential_4/lambda_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2+
)sequential_4/lambda_4/strided_slice/stack│
+sequential_4/lambda_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2-
+sequential_4/lambda_4/strided_slice/stack_1│
+sequential_4/lambda_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2-
+sequential_4/lambda_4/strided_slice/stack_2в
#sequential_4/lambda_4/strided_sliceStridedSliceinputs2sequential_4/lambda_4/strided_slice/stack:output:04sequential_4/lambda_4/strided_slice/stack_1:output:04sequential_4/lambda_4/strided_slice/stack_2:output:0*
Index0*
T0*/
_output_shapes
:         KK*

begin_mask*
end_mask2%
#sequential_4/lambda_4/strided_sliceП
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpс
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1љ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpќ
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1┬
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3,sequential_4/lambda_4/strided_slice:output:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         KK:::::*
epsilon%oЃ:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3┌
,sequential_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_12/Conv2D/ReadVariableOpЎ
sequential_4/conv2d_12/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:04sequential_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK *
paddingSAME*
strides
2
sequential_4/conv2d_12/Conv2DЛ
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpС
sequential_4/conv2d_12/BiasAddBiasAdd&sequential_4/conv2d_12/Conv2D:output:05sequential_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         KK 2 
sequential_4/conv2d_12/BiasAddЦ
sequential_4/conv2d_12/ReluRelu'sequential_4/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:         KK 2
sequential_4/conv2d_12/Reluы
%sequential_4/max_pooling2d_12/MaxPoolMaxPool)sequential_4/conv2d_12/Relu:activations:0*/
_output_shapes
:         %% *
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_12/MaxPool█
,sequential_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
: ђ*
dtype02.
,sequential_4/conv2d_13/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_13/Conv2DConv2D.sequential_4/max_pooling2d_12/MaxPool:output:04sequential_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ*
paddingSAME*
strides
2
sequential_4/conv2d_13/Conv2Dм
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpт
sequential_4/conv2d_13/BiasAddBiasAdd&sequential_4/conv2d_13/Conv2D:output:05sequential_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%ђ2 
sequential_4/conv2d_13/BiasAddд
sequential_4/conv2d_13/ReluRelu'sequential_4/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         %%ђ2
sequential_4/conv2d_13/ReluЫ
%sequential_4/max_pooling2d_13/MaxPoolMaxPool)sequential_4/conv2d_13/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_13/MaxPool▄
,sequential_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_4/conv2d_14/Conv2D/ReadVariableOpЉ
sequential_4/conv2d_14/Conv2DConv2D.sequential_4/max_pooling2d_13/MaxPool:output:04sequential_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_4/conv2d_14/Conv2Dм
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpт
sequential_4/conv2d_14/BiasAddBiasAdd&sequential_4/conv2d_14/Conv2D:output:05sequential_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_4/conv2d_14/BiasAddд
sequential_4/conv2d_14/ReluRelu'sequential_4/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
sequential_4/conv2d_14/ReluЫ
%sequential_4/max_pooling2d_14/MaxPoolMaxPool)sequential_4/conv2d_14/Relu:activations:0*0
_output_shapes
:         		ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_4/max_pooling2d_14/MaxPool╗
 sequential_4/dropout_12/IdentityIdentity.sequential_4/max_pooling2d_14/MaxPool:output:0*
T0*0
_output_shapes
:         		ђ2"
 sequential_4/dropout_12/IdentityЇ
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"     Q  2
sequential_4/flatten_4/ConstЛ
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*)
_output_shapes
:         ђб2 
sequential_4/flatten_4/Reshapeм
+sequential_4/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype02-
+sequential_4/dense_12/MatMul/ReadVariableOpО
sequential_4/dense_12/MatMulMatMul'sequential_4/flatten_4/Reshape:output:03sequential_4/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/MatMul¤
,sequential_4/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_12/BiasAdd/ReadVariableOp┌
sequential_4/dense_12/BiasAddBiasAdd&sequential_4/dense_12/MatMul:product:04sequential_4/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/BiasAddЏ
sequential_4/dense_12/ReluRelu&sequential_4/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_12/ReluГ
 sequential_4/dropout_13/IdentityIdentity(sequential_4/dense_12/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_13/IdentityЛ
+sequential_4/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02-
+sequential_4/dense_13/MatMul/ReadVariableOp┘
sequential_4/dense_13/MatMulMatMul)sequential_4/dropout_13/Identity:output:03sequential_4/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/MatMul¤
,sequential_4/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02.
,sequential_4/dense_13/BiasAdd/ReadVariableOp┌
sequential_4/dense_13/BiasAddBiasAdd&sequential_4/dense_13/MatMul:product:04sequential_4/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/BiasAddЏ
sequential_4/dense_13/ReluRelu&sequential_4/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_4/dense_13/ReluГ
 sequential_4/dropout_14/IdentityIdentity(sequential_4/dense_13/Relu:activations:0*
T0*(
_output_shapes
:         ђ2"
 sequential_4/dropout_14/IdentityЕ
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_14/MatMul/ReadVariableOp▒
dense_14/MatMulMatMul)sequential_4/dropout_14/Identity:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/MatMulД
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOpЦ
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_14/BiasAdd|
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_14/SoftmaxТ
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulя
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_12_matmul_readvariableop_resource*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mulП
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4sequential_4_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulЏ
IdentityIdentitydense_14/Softmax:softmax:03^conv2d_12/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_4/batch_normalization_4/ReadVariableOp4^sequential_4/batch_normalization_4/ReadVariableOp_1.^sequential_4/conv2d_12/BiasAdd/ReadVariableOp-^sequential_4/conv2d_12/Conv2D/ReadVariableOp.^sequential_4/conv2d_13/BiasAdd/ReadVariableOp-^sequential_4/conv2d_13/Conv2D/ReadVariableOp.^sequential_4/conv2d_14/BiasAdd/ReadVariableOp-^sequential_4/conv2d_14/Conv2D/ReadVariableOp-^sequential_4/dense_12/BiasAdd/ReadVariableOp,^sequential_4/dense_12/MatMul/ReadVariableOp-^sequential_4/dense_13/BiasAdd/ReadVariableOp,^sequential_4/dense_13/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2ѕ
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_4/batch_normalization_4/ReadVariableOp1sequential_4/batch_normalization_4/ReadVariableOp2j
3sequential_4/batch_normalization_4/ReadVariableOp_13sequential_4/batch_normalization_4/ReadVariableOp_12^
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp-sequential_4/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_12/Conv2D/ReadVariableOp,sequential_4/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp-sequential_4/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_13/Conv2D/ReadVariableOp,sequential_4/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp-sequential_4/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_14/Conv2D/ReadVariableOp,sequential_4/conv2d_14/Conv2D/ReadVariableOp2\
,sequential_4/dense_12/BiasAdd/ReadVariableOp,sequential_4/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_12/MatMul/ReadVariableOp+sequential_4/dense_12/MatMul/ReadVariableOp2\
,sequential_4/dense_13/BiasAdd/ReadVariableOp,sequential_4/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_13/MatMul/ReadVariableOp+sequential_4/dense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
ш1
ф
?__inference_CNN_layer_call_and_return_conditional_losses_702863

inputs!
sequential_4_702798:!
sequential_4_702800:!
sequential_4_702802:!
sequential_4_702804:-
sequential_4_702806: !
sequential_4_702808: .
sequential_4_702810: ђ"
sequential_4_702812:	ђ/
sequential_4_702814:ђђ"
sequential_4_702816:	ђ(
sequential_4_702818:ђбђ"
sequential_4_702820:	ђ'
sequential_4_702822:
ђђ"
sequential_4_702824:	ђ"
dense_14_702839:	ђ
dense_14_702841:
identityѕб2conv2d_12/kernel/Regularizer/Square/ReadVariableOpб1dense_12/kernel/Regularizer/Square/ReadVariableOpб1dense_13/kernel/Regularizer/Square/ReadVariableOpб dense_14/StatefulPartitionedCallб$sequential_4/StatefulPartitionedCall┬
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinputssequential_4_702798sequential_4_702800sequential_4_702802sequential_4_702804sequential_4_702806sequential_4_702808sequential_4_702810sequential_4_702812sequential_4_702814sequential_4_702816sequential_4_702818sequential_4_702820sequential_4_702822sequential_4_702824*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_7022812&
$sequential_4/StatefulPartitionedCall└
 dense_14/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0dense_14_702839dense_14_702841*
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
GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_7028382"
 dense_14/StatefulPartitionedCall─
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_702806*&
_output_shapes
: *
dtype024
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp┴
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_12/kernel/Regularizer/SquareА
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_12/kernel/Regularizer/Const┬
 conv2d_12/kernel/Regularizer/SumSum'conv2d_12/kernel/Regularizer/Square:y:0+conv2d_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/SumЇ
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2$
"conv2d_12/kernel/Regularizer/mul/x─
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_12/kernel/Regularizer/mulй
1dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_702818*!
_output_shapes
:ђбђ*
dtype023
1dense_12/kernel/Regularizer/Square/ReadVariableOp╣
"dense_12/kernel/Regularizer/SquareSquare9dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*!
_output_shapes
:ђбђ2$
"dense_12/kernel/Regularizer/SquareЌ
!dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_12/kernel/Regularizer/ConstЙ
dense_12/kernel/Regularizer/SumSum&dense_12/kernel/Regularizer/Square:y:0*dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/SumІ
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_12/kernel/Regularizer/mul/x└
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0(dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_12/kernel/Regularizer/mul╝
1dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_702822* 
_output_shapes
:
ђђ*
dtype023
1dense_13/kernel/Regularizer/Square/ReadVariableOpИ
"dense_13/kernel/Regularizer/SquareSquare9dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ђђ2$
"dense_13/kernel/Regularizer/SquareЌ
!dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_13/kernel/Regularizer/ConstЙ
dense_13/kernel/Regularizer/SumSum&dense_13/kernel/Regularizer/Square:y:0*dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/SumІ
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2#
!dense_13/kernel/Regularizer/mul/x└
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0(dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_13/kernel/Regularizer/mulС
IdentityIdentity)dense_14/StatefulPartitionedCall:output:03^conv2d_12/kernel/Regularizer/Square/ReadVariableOp2^dense_12/kernel/Regularizer/Square/ReadVariableOp2^dense_13/kernel/Regularizer/Square/ReadVariableOp!^dense_14/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:         KK: : : : : : : : : : : : : : : : 2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_12/kernel/Regularizer/Square/ReadVariableOp1dense_12/kernel/Regularizer/Square/ReadVariableOp2f
1dense_13/kernel/Regularizer/Square/ReadVariableOp1dense_13/kernel/Regularizer/Square/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall:W S
/
_output_shapes
:         KK
 
_user_specified_nameinputs
Ш
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_702404

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         		ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         		ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/yК
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         		ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         		ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         		ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         		ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         		ђ:X T
0
_output_shapes
:         		ђ
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЪ
C
input_18
serving_default_input_1:0         KK<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:ЗЃ
■


h2ptjl
_output
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+ж&call_and_return_all_conditional_losses
Ж_default_save_signature
в__call__
	Вcall"ї	
_tf_keras_modelЫ{"name": "CNN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CNN", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CNN"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
└h
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
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
regularization_losses
	variables
trainable_variables
	keras_api
+ь&call_and_return_all_conditional_losses
Ь__call__"┘d
_tf_keras_sequential║d{"name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_4_input"}}, {"class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_14", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}]}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 75, 75, 2]}, "float32", "lambda_4_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 75, 75, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_4_input"}, "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_14", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}]}}}
О

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
+№&call_and_return_all_conditional_losses
­__call__"░
_tf_keras_layerќ{"name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
в
!iter

"beta_1

#beta_2
	$decay
%learning_ratem═m╬&m¤'mл*mЛ+mм,mМ-mн.mН/mо0mО1mп2m┘3m┌v█v▄&vП'vя*v▀+vЯ,vр-vР.vс/vС0vт1vТ2vу3vУ"
	optimizer
 "
trackable_list_wrapper
ќ
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
212
313
14
15"
trackable_list_wrapper
є
&0
'1
*2
+3
,4
-5
.6
/7
08
19
210
311
12
13"
trackable_list_wrapper
╬
regularization_losses
4non_trainable_variables
	variables
5layer_regularization_losses
6metrics

7layers
8layer_metrics
trainable_variables
в__call__
Ж_default_save_signature
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
-
ыserving_default"
signature_map
п
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+Ы&call_and_return_all_conditional_losses
з__call__"К
_tf_keras_layerГ{"name": "lambda_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAGAAAAUwAAAHMgAAAAfABkAGQAhQJkAGQAhQJkAGQAhQJkAGQAhQJm\nBBkAUwCpAU6pAKkB2gF4cgIAAAByAgAAAPofL2hvbWUvc2FtaHVhbmcvTUwvQ05OL21vZGVscy5w\nedoIPGxhbWJkYT4RAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "models", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 3}
─

=axis
	&gamma
'beta
(moving_mean
)moving_variance
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+З&call_and_return_all_conditional_losses
ш__call__"Ь
_tf_keras_layerн{"name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 2}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
б

*kernel
+bias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+Ш&call_and_return_all_conditional_losses
э__call__"ч	
_tf_keras_layerр	{"name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 75, 75, 2]}}
│
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+Э&call_and_return_all_conditional_losses
щ__call__"б
_tf_keras_layerѕ{"name": "max_pooling2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 41}}
о


,kernel
-bias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+Щ&call_and_return_all_conditional_losses
ч__call__"»	
_tf_keras_layerЋ	{"name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 37, 37, 32]}}
│
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+Ч&call_and_return_all_conditional_losses
§__call__"б
_tf_keras_layerѕ{"name": "max_pooling2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
п


.kernel
/bias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
+■&call_and_return_all_conditional_losses
 __call__"▒	
_tf_keras_layerЌ	{"name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 18, 18, 128]}}
│
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
+ђ&call_and_return_all_conditional_losses
Ђ__call__"б
_tf_keras_layerѕ{"name": "max_pooling2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_14", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
Ђ
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
+ѓ&call_and_return_all_conditional_losses
Ѓ__call__"­
_tf_keras_layerо{"name": "dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 22}
ў
^regularization_losses
_	variables
`trainable_variables
a	keras_api
+ё&call_and_return_all_conditional_losses
Ё__call__"Є
_tf_keras_layerь{"name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
ф	

0kernel
1bias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+є&call_and_return_all_conditional_losses
Є__call__"Ѓ
_tf_keras_layerж{"name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 26}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20736}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 20736]}}
Ђ
fregularization_losses
g	variables
htrainable_variables
i	keras_api
+ѕ&call_and_return_all_conditional_losses
Ѕ__call__"­
_tf_keras_layerо{"name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 28}
д	

2kernel
3bias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
+і&call_and_return_all_conditional_losses
І__call__" 
_tf_keras_layerт{"name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
Ђ
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+ї&call_and_return_all_conditional_losses
Ї__call__"­
_tf_keras_layerо{"name": "dropout_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 33}
8
ј0
Ј1
љ2"
trackable_list_wrapper
є
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
212
313"
trackable_list_wrapper
v
&0
'1
*2
+3
,4
-5
.6
/7
08
19
210
311"
trackable_list_wrapper
░
regularization_losses
rnon_trainable_variables
	variables
slayer_regularization_losses
tmetrics

ulayers
vlayer_metrics
trainable_variables
Ь__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
": 	ђ2dense_14/kernel
:2dense_14/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
wnon_trainable_variables
regularization_losses
xlayer_regularization_losses
	variables
ymetrics

zlayers
{layer_metrics
trainable_variables
­__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'2batch_normalization_4/gamma
(:&2batch_normalization_4/beta
1:/ (2!batch_normalization_4/moving_mean
5:3 (2%batch_normalization_4/moving_variance
*:( 2conv2d_12/kernel
: 2conv2d_12/bias
+:) ђ2conv2d_13/kernel
:ђ2conv2d_13/bias
,:*ђђ2conv2d_14/kernel
:ђ2conv2d_14/bias
$:"ђбђ2dense_12/kernel
:ђ2dense_12/bias
#:!
ђђ2dense_13/kernel
:ђ2dense_13/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
│
~non_trainable_variables
9regularization_losses
layer_regularization_losses
:	variables
ђmetrics
Ђlayers
ѓlayer_metrics
;trainable_variables
з__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
х
Ѓnon_trainable_variables
>regularization_losses
 ёlayer_regularization_losses
?	variables
Ёmetrics
єlayers
Єlayer_metrics
@trainable_variables
ш__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
(
ј0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
х
ѕnon_trainable_variables
Bregularization_losses
 Ѕlayer_regularization_losses
C	variables
іmetrics
Іlayers
їlayer_metrics
Dtrainable_variables
э__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Їnon_trainable_variables
Fregularization_losses
 јlayer_regularization_losses
G	variables
Јmetrics
љlayers
Љlayer_metrics
Htrainable_variables
щ__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
х
њnon_trainable_variables
Jregularization_losses
 Њlayer_regularization_losses
K	variables
ћmetrics
Ћlayers
ќlayer_metrics
Ltrainable_variables
ч__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ќnon_trainable_variables
Nregularization_losses
 ўlayer_regularization_losses
O	variables
Ўmetrics
џlayers
Џlayer_metrics
Ptrainable_variables
§__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
х
юnon_trainable_variables
Rregularization_losses
 Юlayer_regularization_losses
S	variables
ъmetrics
Ъlayers
аlayer_metrics
Ttrainable_variables
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
х
Аnon_trainable_variables
Vregularization_losses
 бlayer_regularization_losses
W	variables
Бmetrics
цlayers
Цlayer_metrics
Xtrainable_variables
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
дnon_trainable_variables
Zregularization_losses
 Дlayer_regularization_losses
[	variables
еmetrics
Еlayers
фlayer_metrics
\trainable_variables
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Фnon_trainable_variables
^regularization_losses
 гlayer_regularization_losses
_	variables
Гmetrics
«layers
»layer_metrics
`trainable_variables
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
(
Ј0"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
х
░non_trainable_variables
bregularization_losses
 ▒layer_regularization_losses
c	variables
▓metrics
│layers
┤layer_metrics
dtrainable_variables
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
хnon_trainable_variables
fregularization_losses
 Хlayer_regularization_losses
g	variables
иmetrics
Иlayers
╣layer_metrics
htrainable_variables
Ѕ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
(
љ0"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
х
║non_trainable_variables
jregularization_losses
 ╗layer_regularization_losses
k	variables
╝metrics
йlayers
Йlayer_metrics
ltrainable_variables
І__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
┐non_trainable_variables
nregularization_losses
 └layer_regularization_losses
o	variables
┴metrics
┬layers
├layer_metrics
ptrainable_variables
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
є
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
п

─total

┼count
к	variables
К	keras_api"Ю
_tf_keras_metricѓ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}
Џ

╚total

╔count
╩
_fn_kwargs
╦	variables
╠	keras_api"¤
_tf_keras_metric┤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 1}
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
(0
)1"
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
(
ј0"
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
(
Ј0"
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
(
љ0"
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
:  (2total
:  (2count
0
─0
┼1"
trackable_list_wrapper
.
к	variables"
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
':%	ђ2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
.:,2"Adam/batch_normalization_4/gamma/m
-:+2!Adam/batch_normalization_4/beta/m
/:- 2Adam/conv2d_12/kernel/m
!: 2Adam/conv2d_12/bias/m
0:. ђ2Adam/conv2d_13/kernel/m
": ђ2Adam/conv2d_13/bias/m
1:/ђђ2Adam/conv2d_14/kernel/m
": ђ2Adam/conv2d_14/bias/m
):'ђбђ2Adam/dense_12/kernel/m
!:ђ2Adam/dense_12/bias/m
(:&
ђђ2Adam/dense_13/kernel/m
!:ђ2Adam/dense_13/bias/m
':%	ђ2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
.:,2"Adam/batch_normalization_4/gamma/v
-:+2!Adam/batch_normalization_4/beta/v
/:- 2Adam/conv2d_12/kernel/v
!: 2Adam/conv2d_12/bias/v
0:. ђ2Adam/conv2d_13/kernel/v
": ђ2Adam/conv2d_13/bias/v
1:/ђђ2Adam/conv2d_14/kernel/v
": ђ2Adam/conv2d_14/bias/v
):'ђбђ2Adam/dense_12/kernel/v
!:ђ2Adam/dense_12/bias/v
(:&
ђђ2Adam/dense_13/kernel/v
!:ђ2Adam/dense_13/bias/v
Й2╗
?__inference_CNN_layer_call_and_return_conditional_losses_703340
?__inference_CNN_layer_call_and_return_conditional_losses_703451
?__inference_CNN_layer_call_and_return_conditional_losses_703541
?__inference_CNN_layer_call_and_return_conditional_losses_703652┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
у2С
!__inference__wrapped_model_701921Й
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *.б+
)і&
input_1         KK
м2¤
$__inference_CNN_layer_call_fn_703689
$__inference_CNN_layer_call_fn_703726
$__inference_CNN_layer_call_fn_703763
$__inference_CNN_layer_call_fn_703800┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ё2Ђ
__inference_call_625998
__inference_call_626070
__inference_call_626142│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
H__inference_sequential_4_layer_call_and_return_conditional_losses_703901
H__inference_sequential_4_layer_call_and_return_conditional_losses_704005
H__inference_sequential_4_layer_call_and_return_conditional_losses_704088
H__inference_sequential_4_layer_call_and_return_conditional_losses_704192└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ѓ2 
-__inference_sequential_4_layer_call_fn_704225
-__inference_sequential_4_layer_call_fn_704258
-__inference_sequential_4_layer_call_fn_704291
-__inference_sequential_4_layer_call_fn_704324└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
D__inference_dense_14_layer_call_and_return_conditional_losses_704335б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_14_layer_call_fn_704344б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╦B╚
$__inference_signature_wrapper_703250input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
D__inference_lambda_4_layer_call_and_return_conditional_losses_704352
D__inference_lambda_4_layer_call_and_return_conditional_losses_704360└
и▓│
FullArgSpec1
args)џ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsџ

 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ю2Ў
)__inference_lambda_4_layer_call_fn_704365
)__inference_lambda_4_layer_call_fn_704370└
и▓│
FullArgSpec1
args)џ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsџ

 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704388
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704406
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704424
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704442┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
џ2Ќ
6__inference_batch_normalization_4_layer_call_fn_704455
6__inference_batch_normalization_4_layer_call_fn_704468
6__inference_batch_normalization_4_layer_call_fn_704481
6__inference_batch_normalization_4_layer_call_fn_704494┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
№2В
E__inference_conv2d_12_layer_call_and_return_conditional_losses_704517б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_12_layer_call_fn_704526б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┤2▒
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_702053Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ў2ќ
1__inference_max_pooling2d_12_layer_call_fn_702059Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
№2В
E__inference_conv2d_13_layer_call_and_return_conditional_losses_704537б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_13_layer_call_fn_704546б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┤2▒
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_702065Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ў2ќ
1__inference_max_pooling2d_13_layer_call_fn_702071Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
№2В
E__inference_conv2d_14_layer_call_and_return_conditional_losses_704557б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_14_layer_call_fn_704566б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┤2▒
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_702077Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Ў2ќ
1__inference_max_pooling2d_14_layer_call_fn_702083Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
╩2К
F__inference_dropout_12_layer_call_and_return_conditional_losses_704571
F__inference_dropout_12_layer_call_and_return_conditional_losses_704583┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ћ2Љ
+__inference_dropout_12_layer_call_fn_704588
+__inference_dropout_12_layer_call_fn_704593┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
№2В
E__inference_flatten_4_layer_call_and_return_conditional_losses_704599б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_flatten_4_layer_call_fn_704604б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_12_layer_call_and_return_conditional_losses_704627б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_12_layer_call_fn_704636б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╩2К
F__inference_dropout_13_layer_call_and_return_conditional_losses_704641
F__inference_dropout_13_layer_call_and_return_conditional_losses_704653┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ћ2Љ
+__inference_dropout_13_layer_call_fn_704658
+__inference_dropout_13_layer_call_fn_704663┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
D__inference_dense_13_layer_call_and_return_conditional_losses_704686б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_13_layer_call_fn_704695б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╩2К
F__inference_dropout_14_layer_call_and_return_conditional_losses_704700
F__inference_dropout_14_layer_call_and_return_conditional_losses_704712┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ћ2Љ
+__inference_dropout_14_layer_call_fn_704717
+__inference_dropout_14_layer_call_fn_704722┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
│2░
__inference_loss_fn_0_704733Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
│2░
__inference_loss_fn_1_704744Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
│2░
__inference_loss_fn_2_704755Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б ╣
?__inference_CNN_layer_call_and_return_conditional_losses_703340v&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p 
ф "%б"
і
0         
џ ╣
?__inference_CNN_layer_call_and_return_conditional_losses_703451v&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p
ф "%б"
і
0         
џ ║
?__inference_CNN_layer_call_and_return_conditional_losses_703541w&'()*+,-./0123<б9
2б/
)і&
input_1         KK
p 
ф "%б"
і
0         
џ ║
?__inference_CNN_layer_call_and_return_conditional_losses_703652w&'()*+,-./0123<б9
2б/
)і&
input_1         KK
p
ф "%б"
і
0         
џ њ
$__inference_CNN_layer_call_fn_703689j&'()*+,-./0123<б9
2б/
)і&
input_1         KK
p 
ф "і         Љ
$__inference_CNN_layer_call_fn_703726i&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p 
ф "і         Љ
$__inference_CNN_layer_call_fn_703763i&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p
ф "і         њ
$__inference_CNN_layer_call_fn_703800j&'()*+,-./0123<б9
2б/
)і&
input_1         KK
p
ф "і         Д
!__inference__wrapped_model_701921Ђ&'()*+,-./01238б5
.б+
)і&
input_1         KK
ф "3ф0
.
output_1"і
output_1         В
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704388ќ&'()MбJ
Cб@
:і7
inputs+                           
p 
ф "?б<
5і2
0+                           
џ В
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704406ќ&'()MбJ
Cб@
:і7
inputs+                           
p
ф "?б<
5і2
0+                           
џ К
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704424r&'();б8
1б.
(і%
inputs         KK
p 
ф "-б*
#і 
0         KK
џ К
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_704442r&'();б8
1б.
(і%
inputs         KK
p
ф "-б*
#і 
0         KK
џ ─
6__inference_batch_normalization_4_layer_call_fn_704455Ѕ&'()MбJ
Cб@
:і7
inputs+                           
p 
ф "2і/+                           ─
6__inference_batch_normalization_4_layer_call_fn_704468Ѕ&'()MбJ
Cб@
:і7
inputs+                           
p
ф "2і/+                           Ъ
6__inference_batch_normalization_4_layer_call_fn_704481e&'();б8
1б.
(і%
inputs         KK
p 
ф " і         KKЪ
6__inference_batch_normalization_4_layer_call_fn_704494e&'();б8
1б.
(і%
inputs         KK
p
ф " і         KKt
__inference_call_625998Y&'()*+,-./01233б0
)б&
 і
inputsђKK
p
ф "і	ђt
__inference_call_626070Y&'()*+,-./01233б0
)б&
 і
inputsђKK
p 
ф "і	ђё
__inference_call_626142i&'()*+,-./0123;б8
1б.
(і%
inputs         KK
p 
ф "і         х
E__inference_conv2d_12_layer_call_and_return_conditional_losses_704517l*+7б4
-б*
(і%
inputs         KK
ф "-б*
#і 
0         KK 
џ Ї
*__inference_conv2d_12_layer_call_fn_704526_*+7б4
-б*
(і%
inputs         KK
ф " і         KK Х
E__inference_conv2d_13_layer_call_and_return_conditional_losses_704537m,-7б4
-б*
(і%
inputs         %% 
ф ".б+
$і!
0         %%ђ
џ ј
*__inference_conv2d_13_layer_call_fn_704546`,-7б4
-б*
(і%
inputs         %% 
ф "!і         %%ђи
E__inference_conv2d_14_layer_call_and_return_conditional_losses_704557n./8б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ Ј
*__inference_conv2d_14_layer_call_fn_704566a./8б5
.б+
)і&
inputs         ђ
ф "!і         ђД
D__inference_dense_12_layer_call_and_return_conditional_losses_704627_011б.
'б$
"і
inputs         ђб
ф "&б#
і
0         ђ
џ 
)__inference_dense_12_layer_call_fn_704636R011б.
'б$
"і
inputs         ђб
ф "і         ђд
D__inference_dense_13_layer_call_and_return_conditional_losses_704686^230б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ~
)__inference_dense_13_layer_call_fn_704695Q230б-
&б#
!і
inputs         ђ
ф "і         ђЦ
D__inference_dense_14_layer_call_and_return_conditional_losses_704335]0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ }
)__inference_dense_14_layer_call_fn_704344P0б-
&б#
!і
inputs         ђ
ф "і         И
F__inference_dropout_12_layer_call_and_return_conditional_losses_704571n<б9
2б/
)і&
inputs         		ђ
p 
ф ".б+
$і!
0         		ђ
џ И
F__inference_dropout_12_layer_call_and_return_conditional_losses_704583n<б9
2б/
)і&
inputs         		ђ
p
ф ".б+
$і!
0         		ђ
џ љ
+__inference_dropout_12_layer_call_fn_704588a<б9
2б/
)і&
inputs         		ђ
p 
ф "!і         		ђљ
+__inference_dropout_12_layer_call_fn_704593a<б9
2б/
)і&
inputs         		ђ
p
ф "!і         		ђе
F__inference_dropout_13_layer_call_and_return_conditional_losses_704641^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ е
F__inference_dropout_13_layer_call_and_return_conditional_losses_704653^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ђ
+__inference_dropout_13_layer_call_fn_704658Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђђ
+__inference_dropout_13_layer_call_fn_704663Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђе
F__inference_dropout_14_layer_call_and_return_conditional_losses_704700^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ е
F__inference_dropout_14_layer_call_and_return_conditional_losses_704712^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ ђ
+__inference_dropout_14_layer_call_fn_704717Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђђ
+__inference_dropout_14_layer_call_fn_704722Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђг
E__inference_flatten_4_layer_call_and_return_conditional_losses_704599c8б5
.б+
)і&
inputs         		ђ
ф "'б$
і
0         ђб
џ ё
*__inference_flatten_4_layer_call_fn_704604V8б5
.б+
)і&
inputs         		ђ
ф "і         ђбИ
D__inference_lambda_4_layer_call_and_return_conditional_losses_704352p?б<
5б2
(і%
inputs         KK

 
p 
ф "-б*
#і 
0         KK
џ И
D__inference_lambda_4_layer_call_and_return_conditional_losses_704360p?б<
5б2
(і%
inputs         KK

 
p
ф "-б*
#і 
0         KK
џ љ
)__inference_lambda_4_layer_call_fn_704365c?б<
5б2
(і%
inputs         KK

 
p 
ф " і         KKљ
)__inference_lambda_4_layer_call_fn_704370c?б<
5б2
(і%
inputs         KK

 
p
ф " і         KK;
__inference_loss_fn_0_704733*б

б 
ф "і ;
__inference_loss_fn_1_7047440б

б 
ф "і ;
__inference_loss_fn_2_7047552б

б 
ф "і №
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_702053ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_12_layer_call_fn_702059ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_702065ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_13_layer_call_fn_702071ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    №
L__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_702077ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_14_layer_call_fn_702083ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ┼
H__inference_sequential_4_layer_call_and_return_conditional_losses_703901y&'()*+,-./0123?б<
5б2
(і%
inputs         KK
p 

 
ф "&б#
і
0         ђ
џ ┼
H__inference_sequential_4_layer_call_and_return_conditional_losses_704005y&'()*+,-./0123?б<
5б2
(і%
inputs         KK
p

 
ф "&б#
і
0         ђ
џ ╬
H__inference_sequential_4_layer_call_and_return_conditional_losses_704088Ђ&'()*+,-./0123GбD
=б:
0і-
lambda_4_input         KK
p 

 
ф "&б#
і
0         ђ
џ ╬
H__inference_sequential_4_layer_call_and_return_conditional_losses_704192Ђ&'()*+,-./0123GбD
=б:
0і-
lambda_4_input         KK
p

 
ф "&б#
і
0         ђ
џ Ц
-__inference_sequential_4_layer_call_fn_704225t&'()*+,-./0123GбD
=б:
0і-
lambda_4_input         KK
p 

 
ф "і         ђЮ
-__inference_sequential_4_layer_call_fn_704258l&'()*+,-./0123?б<
5б2
(і%
inputs         KK
p 

 
ф "і         ђЮ
-__inference_sequential_4_layer_call_fn_704291l&'()*+,-./0123?б<
5б2
(і%
inputs         KK
p

 
ф "і         ђЦ
-__inference_sequential_4_layer_call_fn_704324t&'()*+,-./0123GбD
=б:
0і-
lambda_4_input         KK
p

 
ф "і         ђх
$__inference_signature_wrapper_703250ї&'()*+,-./0123Cб@
б 
9ф6
4
input_1)і&
input_1         KK"3ф0
.
output_1"і
output_1         