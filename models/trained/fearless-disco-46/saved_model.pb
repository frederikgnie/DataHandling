н”

ЊП
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
÷
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
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
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28ь°	
В
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:*
dtype0
Ж
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:	*
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:	*
dtype0
Ж
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
:	*
dtype0
r
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
:*
dtype0
Ж
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:	*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:	*
dtype0
Ж
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv3d_4/kernel

#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel**
_output_shapes
:	*
dtype0
r
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:*
dtype0
Ж
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
:*
dtype0
r
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
:*
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
Р
Adam/conv3d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/m
Й
(Adam/conv3d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/m**
_output_shapes
:*
dtype0
|
Adam/conv3d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv3d/bias/m
u
&Adam/conv3d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_1/kernel/m
Н
*Adam/conv3d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/m**
_output_shapes
:	*
dtype0
А
Adam/conv3d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv3d_1/bias/m
y
(Adam/conv3d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/m*
_output_shapes
:	*
dtype0
Ф
Adam/conv3d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_2/kernel/m
Н
*Adam/conv3d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/m**
_output_shapes
:	*
dtype0
А
Adam/conv3d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_2/bias/m
y
(Adam/conv3d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_3/kernel/m
Н
*Adam/conv3d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/m**
_output_shapes
:	*
dtype0
А
Adam/conv3d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv3d_3/bias/m
y
(Adam/conv3d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/m*
_output_shapes
:	*
dtype0
Ф
Adam/conv3d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_4/kernel/m
Н
*Adam/conv3d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/m**
_output_shapes
:	*
dtype0
А
Adam/conv3d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_4/bias/m
y
(Adam/conv3d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/m
Н
*Adam/conv3d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/m**
_output_shapes
:*
dtype0
А
Adam/conv3d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_5/bias/m
y
(Adam/conv3d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/m*
_output_shapes
:*
dtype0
Р
Adam/conv3d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/v
Й
(Adam/conv3d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/v**
_output_shapes
:*
dtype0
|
Adam/conv3d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv3d/bias/v
u
&Adam/conv3d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_1/kernel/v
Н
*Adam/conv3d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/v**
_output_shapes
:	*
dtype0
А
Adam/conv3d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv3d_1/bias/v
y
(Adam/conv3d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/v*
_output_shapes
:	*
dtype0
Ф
Adam/conv3d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_2/kernel/v
Н
*Adam/conv3d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/v**
_output_shapes
:	*
dtype0
А
Adam/conv3d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_2/bias/v
y
(Adam/conv3d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_3/kernel/v
Н
*Adam/conv3d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/v**
_output_shapes
:	*
dtype0
А
Adam/conv3d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv3d_3/bias/v
y
(Adam/conv3d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/v*
_output_shapes
:	*
dtype0
Ф
Adam/conv3d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_4/kernel/v
Н
*Adam/conv3d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/v**
_output_shapes
:	*
dtype0
А
Adam/conv3d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_4/bias/v
y
(Adam/conv3d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/v
Н
*Adam/conv3d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/v**
_output_shapes
:*
dtype0
А
Adam/conv3d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_5/bias/v
y
(Adam/conv3d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
г?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ю?
valueФ?BС? BК?
х
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
R
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
Щ
7iter

8beta_1

9beta_2
	:decay
;learning_ratemimjmkmlmmmn%mo&mp+mq,mr1ms2mtvuvvvwvxvyvz%v{&v|+v},v~1v2vА
V
0
1
2
3
4
5
%6
&7
+8
,9
110
211
V
0
1
2
3
4
5
%6
&7
+8
,9
110
211
 
≠
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics

	variables
trainable_variables
regularization_losses
 
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
≠
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
!	variables
"trainable_variables
#regularization_losses
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
≠
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
'	variables
(trainable_variables
)regularization_losses
[Y
VARIABLE_VALUEconv3d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
≠
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
-	variables
.trainable_variables
/regularization_losses
[Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
≠
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
3	variables
4trainable_variables
5regularization_losses
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
 
8
0
1
2
3
4
5
6
7

d0
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
4
	etotal
	fcount
g	variables
h	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

g	variables
|z
VARIABLE_VALUEAdam/conv3d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Т
serving_default_input_1Placeholder*3
_output_shapes!
:€€€€€€€€€   *
dtype0*(
shape:€€€€€€€€€   
Х
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_472055
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
‘
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv3d/kernel/m/Read/ReadVariableOp&Adam/conv3d/bias/m/Read/ReadVariableOp*Adam/conv3d_1/kernel/m/Read/ReadVariableOp(Adam/conv3d_1/bias/m/Read/ReadVariableOp*Adam/conv3d_2/kernel/m/Read/ReadVariableOp(Adam/conv3d_2/bias/m/Read/ReadVariableOp*Adam/conv3d_3/kernel/m/Read/ReadVariableOp(Adam/conv3d_3/bias/m/Read/ReadVariableOp*Adam/conv3d_4/kernel/m/Read/ReadVariableOp(Adam/conv3d_4/bias/m/Read/ReadVariableOp*Adam/conv3d_5/kernel/m/Read/ReadVariableOp(Adam/conv3d_5/bias/m/Read/ReadVariableOp(Adam/conv3d/kernel/v/Read/ReadVariableOp&Adam/conv3d/bias/v/Read/ReadVariableOp*Adam/conv3d_1/kernel/v/Read/ReadVariableOp(Adam/conv3d_1/bias/v/Read/ReadVariableOp*Adam/conv3d_2/kernel/v/Read/ReadVariableOp(Adam/conv3d_2/bias/v/Read/ReadVariableOp*Adam/conv3d_3/kernel/v/Read/ReadVariableOp(Adam/conv3d_3/bias/v/Read/ReadVariableOp*Adam/conv3d_4/kernel/v/Read/ReadVariableOp(Adam/conv3d_4/bias/v/Read/ReadVariableOp*Adam/conv3d_5/kernel/v/Read/ReadVariableOp(Adam/conv3d_5/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_472538
у
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv3d/kernel/mAdam/conv3d/bias/mAdam/conv3d_1/kernel/mAdam/conv3d_1/bias/mAdam/conv3d_2/kernel/mAdam/conv3d_2/bias/mAdam/conv3d_3/kernel/mAdam/conv3d_3/bias/mAdam/conv3d_4/kernel/mAdam/conv3d_4/bias/mAdam/conv3d_5/kernel/mAdam/conv3d_5/bias/mAdam/conv3d/kernel/vAdam/conv3d/bias/vAdam/conv3d_1/kernel/vAdam/conv3d_1/bias/vAdam/conv3d_2/kernel/vAdam/conv3d_2/bias/vAdam/conv3d_3/kernel/vAdam/conv3d_3/bias/vAdam/conv3d_4/kernel/vAdam/conv3d_4/bias/vAdam/conv3d_5/kernel/vAdam/conv3d_5/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_472677По
¶
Б
D__inference_conv3d_1_layer_call_and_return_conditional_losses_472275

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
∆

Б
D__inference_conv3d_5_layer_call_and_return_conditional_losses_472378

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
Х4
С
A__inference_model_layer_call_and_return_conditional_losses_472018
input_1+
conv3d_471977:
conv3d_471979:-
conv3d_1_471982:	
conv3d_1_471984:	-
conv3d_2_471987:	
conv3d_2_471989:-
conv3d_3_472001:	
conv3d_3_472003:	-
conv3d_4_472006:	
conv3d_4_472008:-
conv3d_5_472011:
conv3d_5_472013:
identity

identity_1ИҐconv3d/StatefulPartitionedCallҐ conv3d_1/StatefulPartitionedCallҐ conv3d_2/StatefulPartitionedCallҐ conv3d_3/StatefulPartitionedCallҐ conv3d_4/StatefulPartitionedCallҐ conv3d_5/StatefulPartitionedCallш
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_471977conv3d_471979*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_471572†
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_471982conv3d_1_471984*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_471589Ґ
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_471987conv3d_2_471989*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_471606И
'activity_regularization/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471750т
;activity_regularization/ActivityRegularizer/PartitionedCallPartitionedCall0activity_regularization/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_activity_regularization_activity_regularizer_471554С
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:Й
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskђ
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: џ
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ©
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_3_472001conv3d_3_472003*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_471637Ґ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_472006conv3d_4_472008*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_471654Ґ
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_472011conv3d_5_472013*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_471670Д
IdentityIdentity)conv3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ц
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€   
!
_user_specified_name	input_1
Б
Ґ
)__inference_conv3d_3_layer_call_fn_472328

inputs%
unknown:	
	unknown_0:	
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_471637{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
¶
Б
D__inference_conv3d_4_layer_call_and_return_conditional_losses_472359

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   	
 
_user_specified_nameinputs
љ≠
ћ
"__inference__traced_restore_472677
file_prefix<
assignvariableop_conv3d_kernel:,
assignvariableop_1_conv3d_bias:@
"assignvariableop_2_conv3d_1_kernel:	.
 assignvariableop_3_conv3d_1_bias:	@
"assignvariableop_4_conv3d_2_kernel:	.
 assignvariableop_5_conv3d_2_bias:@
"assignvariableop_6_conv3d_3_kernel:	.
 assignvariableop_7_conv3d_3_bias:	@
"assignvariableop_8_conv3d_4_kernel:	.
 assignvariableop_9_conv3d_4_bias:A
#assignvariableop_10_conv3d_5_kernel:/
!assignvariableop_11_conv3d_5_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: F
(assignvariableop_19_adam_conv3d_kernel_m:4
&assignvariableop_20_adam_conv3d_bias_m:H
*assignvariableop_21_adam_conv3d_1_kernel_m:	6
(assignvariableop_22_adam_conv3d_1_bias_m:	H
*assignvariableop_23_adam_conv3d_2_kernel_m:	6
(assignvariableop_24_adam_conv3d_2_bias_m:H
*assignvariableop_25_adam_conv3d_3_kernel_m:	6
(assignvariableop_26_adam_conv3d_3_bias_m:	H
*assignvariableop_27_adam_conv3d_4_kernel_m:	6
(assignvariableop_28_adam_conv3d_4_bias_m:H
*assignvariableop_29_adam_conv3d_5_kernel_m:6
(assignvariableop_30_adam_conv3d_5_bias_m:F
(assignvariableop_31_adam_conv3d_kernel_v:4
&assignvariableop_32_adam_conv3d_bias_v:H
*assignvariableop_33_adam_conv3d_1_kernel_v:	6
(assignvariableop_34_adam_conv3d_1_bias_v:	H
*assignvariableop_35_adam_conv3d_2_kernel_v:	6
(assignvariableop_36_adam_conv3d_2_bias_v:H
*assignvariableop_37_adam_conv3d_3_kernel_v:	6
(assignvariableop_38_adam_conv3d_3_bias_v:	H
*assignvariableop_39_adam_conv3d_4_kernel_v:	6
(assignvariableop_40_adam_conv3d_4_bias_v:H
*assignvariableop_41_adam_conv3d_5_kernel_v:6
(assignvariableop_42_adam_conv3d_5_bias_v:
identity_44ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ї
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*а
value÷B”,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH»
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B э
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*∆
_output_shapes≥
∞::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv3d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv3d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv3d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv3d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv3d_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv3d_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv3d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv3d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv3d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv3d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv3d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv3d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv3d_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv3d_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv3d_5_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv3d_5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv3d_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv3d_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv3d_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv3d_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv3d_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv3d_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv3d_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv3d_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv3d_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv3d_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv3d_5_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv3d_5_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Б
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: о
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_42AssignVariableOp_422(
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
І
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_472382

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:€€€€€€€€€   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   :[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
г
л
&__inference_model_layer_call_fn_471930
input_1%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€   : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_471872{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€   
!
_user_specified_name	input_1
Т4
Р
A__inference_model_layer_call_and_return_conditional_losses_471872

inputs+
conv3d_471831:
conv3d_471833:-
conv3d_1_471836:	
conv3d_1_471838:	-
conv3d_2_471841:	
conv3d_2_471843:-
conv3d_3_471855:	
conv3d_3_471857:	-
conv3d_4_471860:	
conv3d_4_471862:-
conv3d_5_471865:
conv3d_5_471867:
identity

identity_1ИҐconv3d/StatefulPartitionedCallҐ conv3d_1/StatefulPartitionedCallҐ conv3d_2/StatefulPartitionedCallҐ conv3d_3/StatefulPartitionedCallҐ conv3d_4/StatefulPartitionedCallҐ conv3d_5/StatefulPartitionedCallч
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_471831conv3d_471833*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_471572†
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_471836conv3d_1_471838*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_471589Ґ
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_471841conv3d_2_471843*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_471606И
'activity_regularization/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471750т
;activity_regularization/ActivityRegularizer/PartitionedCallPartitionedCall0activity_regularization/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_activity_regularization_activity_regularizer_471554С
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:Й
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskђ
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: џ
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ©
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_3_471855conv3d_3_471857*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_471637Ґ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_471860conv3d_4_471862*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_471654Ґ
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_471865conv3d_5_471867*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_471670Д
IdentityIdentity)conv3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ц
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
Х4
С
A__inference_model_layer_call_and_return_conditional_losses_471974
input_1+
conv3d_471933:
conv3d_471935:-
conv3d_1_471938:	
conv3d_1_471940:	-
conv3d_2_471943:	
conv3d_2_471945:-
conv3d_3_471957:	
conv3d_3_471959:	-
conv3d_4_471962:	
conv3d_4_471964:-
conv3d_5_471967:
conv3d_5_471969:
identity

identity_1ИҐconv3d/StatefulPartitionedCallҐ conv3d_1/StatefulPartitionedCallҐ conv3d_2/StatefulPartitionedCallҐ conv3d_3/StatefulPartitionedCallҐ conv3d_4/StatefulPartitionedCallҐ conv3d_5/StatefulPartitionedCallш
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_471933conv3d_471935*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_471572†
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_471938conv3d_1_471940*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_471589Ґ
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_471943conv3d_2_471945*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_471606И
'activity_regularization/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471616т
;activity_regularization/ActivityRegularizer/PartitionedCallPartitionedCall0activity_regularization/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_activity_regularization_activity_regularizer_471554С
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:Й
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskђ
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: џ
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ©
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_3_471957conv3d_3_471959*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_471637Ґ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_471962conv3d_4_471964*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_471654Ґ
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_471967conv3d_5_471969*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_471670Д
IdentityIdentity)conv3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ц
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€   
!
_user_specified_name	input_1
†[
–
__inference__traced_save_472538
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop.
*savev2_conv3d_1_kernel_read_readvariableop,
(savev2_conv3d_1_bias_read_readvariableop.
*savev2_conv3d_2_kernel_read_readvariableop,
(savev2_conv3d_2_bias_read_readvariableop.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop.
*savev2_conv3d_4_kernel_read_readvariableop,
(savev2_conv3d_4_bias_read_readvariableop.
*savev2_conv3d_5_kernel_read_readvariableop,
(savev2_conv3d_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv3d_kernel_m_read_readvariableop1
-savev2_adam_conv3d_bias_m_read_readvariableop5
1savev2_adam_conv3d_1_kernel_m_read_readvariableop3
/savev2_adam_conv3d_1_bias_m_read_readvariableop5
1savev2_adam_conv3d_2_kernel_m_read_readvariableop3
/savev2_adam_conv3d_2_bias_m_read_readvariableop5
1savev2_adam_conv3d_3_kernel_m_read_readvariableop3
/savev2_adam_conv3d_3_bias_m_read_readvariableop5
1savev2_adam_conv3d_4_kernel_m_read_readvariableop3
/savev2_adam_conv3d_4_bias_m_read_readvariableop5
1savev2_adam_conv3d_5_kernel_m_read_readvariableop3
/savev2_adam_conv3d_5_bias_m_read_readvariableop3
/savev2_adam_conv3d_kernel_v_read_readvariableop1
-savev2_adam_conv3d_bias_v_read_readvariableop5
1savev2_adam_conv3d_1_kernel_v_read_readvariableop3
/savev2_adam_conv3d_1_bias_v_read_readvariableop5
1savev2_adam_conv3d_2_kernel_v_read_readvariableop3
/savev2_adam_conv3d_2_bias_v_read_readvariableop5
1savev2_adam_conv3d_3_kernel_v_read_readvariableop3
/savev2_adam_conv3d_3_bias_v_read_readvariableop5
1savev2_adam_conv3d_4_kernel_v_read_readvariableop3
/savev2_adam_conv3d_4_bias_v_read_readvariableop5
1savev2_adam_conv3d_5_kernel_v_read_readvariableop3
/savev2_adam_conv3d_5_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
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
: Ј
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*а
value÷B”,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH≈
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B У
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv3d_kernel_m_read_readvariableop-savev2_adam_conv3d_bias_m_read_readvariableop1savev2_adam_conv3d_1_kernel_m_read_readvariableop/savev2_adam_conv3d_1_bias_m_read_readvariableop1savev2_adam_conv3d_2_kernel_m_read_readvariableop/savev2_adam_conv3d_2_bias_m_read_readvariableop1savev2_adam_conv3d_3_kernel_m_read_readvariableop/savev2_adam_conv3d_3_bias_m_read_readvariableop1savev2_adam_conv3d_4_kernel_m_read_readvariableop/savev2_adam_conv3d_4_bias_m_read_readvariableop1savev2_adam_conv3d_5_kernel_m_read_readvariableop/savev2_adam_conv3d_5_bias_m_read_readvariableop/savev2_adam_conv3d_kernel_v_read_readvariableop-savev2_adam_conv3d_bias_v_read_readvariableop1savev2_adam_conv3d_1_kernel_v_read_readvariableop/savev2_adam_conv3d_1_bias_v_read_readvariableop1savev2_adam_conv3d_2_kernel_v_read_readvariableop/savev2_adam_conv3d_2_bias_v_read_readvariableop1savev2_adam_conv3d_3_kernel_v_read_readvariableop/savev2_adam_conv3d_3_bias_v_read_readvariableop1savev2_adam_conv3d_4_kernel_v_read_readvariableop/savev2_adam_conv3d_4_bias_v_read_readvariableop1savev2_adam_conv3d_5_kernel_v_read_readvariableop/savev2_adam_conv3d_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*Я
_input_shapesН
К: :::	:	:	::	:	:	:::: : : : : : : :::	:	:	::	:	:	::::::	:	:	::	:	:	:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:	: 

_output_shapes
:	:0,
*
_output_shapes
:	: 

_output_shapes
::0,
*
_output_shapes
:	: 

_output_shapes
:	:0	,
*
_output_shapes
:	: 


_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:	: 

_output_shapes
:	:0,
*
_output_shapes
:	: 

_output_shapes
::0,
*
_output_shapes
:	: 

_output_shapes
:	:0,
*
_output_shapes
:	: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0 ,
*
_output_shapes
:: !

_output_shapes
::0",
*
_output_shapes
:	: #

_output_shapes
:	:0$,
*
_output_shapes
:	: %

_output_shapes
::0&,
*
_output_shapes
:	: '

_output_shapes
:	:0(,
*
_output_shapes
:	: )

_output_shapes
::0*,
*
_output_shapes
:: +

_output_shapes
::,

_output_shapes
: 
†
V
?__inference_activity_regularization_activity_regularizer_471554
x
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    0
AbsAbsx*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:€€€€€€€€€D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЈQ8I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: F
addAddV2Const:output:0mul:z:0*
T0*
_output_shapes
: >
IdentityIdentityadd:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
∆

Б
D__inference_conv3d_5_layer_call_and_return_conditional_losses_471670

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
»S
’

!__inference__wrapped_model_471539
input_1I
+model_conv3d_conv3d_readvariableop_resource::
,model_conv3d_biasadd_readvariableop_resource:K
-model_conv3d_1_conv3d_readvariableop_resource:	<
.model_conv3d_1_biasadd_readvariableop_resource:	K
-model_conv3d_2_conv3d_readvariableop_resource:	<
.model_conv3d_2_biasadd_readvariableop_resource:K
-model_conv3d_3_conv3d_readvariableop_resource:	<
.model_conv3d_3_biasadd_readvariableop_resource:	K
-model_conv3d_4_conv3d_readvariableop_resource:	<
.model_conv3d_4_biasadd_readvariableop_resource:K
-model_conv3d_5_conv3d_readvariableop_resource:<
.model_conv3d_5_biasadd_readvariableop_resource:
identityИҐ#model/conv3d/BiasAdd/ReadVariableOpҐ"model/conv3d/Conv3D/ReadVariableOpҐ%model/conv3d_1/BiasAdd/ReadVariableOpҐ$model/conv3d_1/Conv3D/ReadVariableOpҐ%model/conv3d_2/BiasAdd/ReadVariableOpҐ$model/conv3d_2/Conv3D/ReadVariableOpҐ%model/conv3d_3/BiasAdd/ReadVariableOpҐ$model/conv3d_3/Conv3D/ReadVariableOpҐ%model/conv3d_4/BiasAdd/ReadVariableOpҐ$model/conv3d_4/Conv3D/ReadVariableOpҐ%model/conv3d_5/BiasAdd/ReadVariableOpҐ$model/conv3d_5/Conv3D/ReadVariableOpЪ
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0є
model/conv3d/Conv3DConv3Dinput_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
М
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0®
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   v
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   Ю
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0’
model/conv3d_1/Conv3DConv3Dmodel/conv3d/Relu:activations:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
Р
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Ѓ
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	z
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	Ю
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0„
model/conv3d_2/Conv3DConv3D!model/conv3d_1/Relu:activations:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
Р
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѓ
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   z
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   |
7model/activity_regularization/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Э
5model/activity_regularization/ActivityRegularizer/AbsAbs!model/conv3d_2/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€   Ц
9model/activity_regularization/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*)
value B"                №
5model/activity_regularization/ActivityRegularizer/SumSum9model/activity_regularization/ActivityRegularizer/Abs:y:0Bmodel/activity_regularization/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: |
7model/activity_regularization/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЈQ8я
5model/activity_regularization/ActivityRegularizer/mulMul@model/activity_regularization/ActivityRegularizer/mul/x:output:0>model/activity_regularization/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: №
5model/activity_regularization/ActivityRegularizer/addAddV2@model/activity_regularization/ActivityRegularizer/Const:output:09model/activity_regularization/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: И
7model/activity_regularization/ActivityRegularizer/ShapeShape!model/conv3d_2/Relu:activations:0*
T0*
_output_shapes
:П
Emodel/activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: С
Gmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:С
Gmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
?model/activity_regularization/ActivityRegularizer/strided_sliceStridedSlice@model/activity_regularization/ActivityRegularizer/Shape:output:0Nmodel/activity_regularization/ActivityRegularizer/strided_slice/stack:output:0Pmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Pmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЄ
6model/activity_regularization/ActivityRegularizer/CastCastHmodel/activity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: №
9model/activity_regularization/ActivityRegularizer/truedivRealDiv9model/activity_regularization/ActivityRegularizer/add:z:0:model/activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ю
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0„
model/conv3d_3/Conv3DConv3D!model/conv3d_2/Relu:activations:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
Р
%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Ѓ
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	z
model/conv3d_3/ReluRelumodel/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	Ю
$model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0„
model/conv3d_4/Conv3DConv3D!model/conv3d_3/Relu:activations:0,model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
Р
%model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѓ
model/conv3d_4/BiasAddBiasAddmodel/conv3d_4/Conv3D:output:0-model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   z
model/conv3d_4/ReluRelumodel/conv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   Ю
$model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0„
model/conv3d_5/Conv3DConv3D!model/conv3d_4/Relu:activations:0,model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
Р
%model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѓ
model/conv3d_5/BiasAddBiasAddmodel/conv3d_5/Conv3D:output:0-model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   z
IdentityIdentitymodel/conv3d_5/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   Ь
NoOpNoOp$^model/conv3d/BiasAdd/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp&^model/conv3d_3/BiasAdd/ReadVariableOp%^model/conv3d_3/Conv3D/ReadVariableOp&^model/conv3d_4/BiasAdd/ReadVariableOp%^model/conv3d_4/Conv3D/ReadVariableOp&^model/conv3d_5/BiasAdd/ReadVariableOp%^model/conv3d_5/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2N
%model/conv3d_3/BiasAdd/ReadVariableOp%model/conv3d_3/BiasAdd/ReadVariableOp2L
$model/conv3d_3/Conv3D/ReadVariableOp$model/conv3d_3/Conv3D/ReadVariableOp2N
%model/conv3d_4/BiasAdd/ReadVariableOp%model/conv3d_4/BiasAdd/ReadVariableOp2L
$model/conv3d_4/Conv3D/ReadVariableOp$model/conv3d_4/Conv3D/ReadVariableOp2N
%model/conv3d_5/BiasAdd/ReadVariableOp%model/conv3d_5/BiasAdd/ReadVariableOp2L
$model/conv3d_5/Conv3D/ReadVariableOp$model/conv3d_5/Conv3D/ReadVariableOp:\ X
3
_output_shapes!
:€€€€€€€€€   
!
_user_specified_name	input_1
а
к
&__inference_model_layer_call_fn_472115

inputs%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€   : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_471872{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
Б
Ґ
)__inference_conv3d_1_layer_call_fn_472264

inputs%
unknown:	
	unknown_0:	
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_471589{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
¶
Б
D__inference_conv3d_1_layer_call_and_return_conditional_losses_471589

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
а
к
&__inference_model_layer_call_fn_472085

inputs%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€   : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_471678{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
Б
Ґ
)__inference_conv3d_2_layer_call_fn_472284

inputs%
unknown:	
	unknown_0:
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_471606{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   	: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   	
 
_user_specified_nameinputs
э
†
'__inference_conv3d_layer_call_fn_472244

inputs%
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_471572{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
§
€
B__inference_conv3d_layer_call_and_return_conditional_losses_472255

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
о
T
8__inference_activity_regularization_layer_call_fn_472305

inputs
identityЌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471750l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   :[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
¶
Б
D__inference_conv3d_3_layer_call_and_return_conditional_losses_472339

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
І
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_472386

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:€€€€€€€€€   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   :[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
Т4
Р
A__inference_model_layer_call_and_return_conditional_losses_471678

inputs+
conv3d_471573:
conv3d_471575:-
conv3d_1_471590:	
conv3d_1_471592:	-
conv3d_2_471607:	
conv3d_2_471609:-
conv3d_3_471638:	
conv3d_3_471640:	-
conv3d_4_471655:	
conv3d_4_471657:-
conv3d_5_471671:
conv3d_5_471673:
identity

identity_1ИҐconv3d/StatefulPartitionedCallҐ conv3d_1/StatefulPartitionedCallҐ conv3d_2/StatefulPartitionedCallҐ conv3d_3/StatefulPartitionedCallҐ conv3d_4/StatefulPartitionedCallҐ conv3d_5/StatefulPartitionedCallч
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_471573conv3d_471575*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_471572†
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_471590conv3d_1_471592*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_471589Ґ
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_471607conv3d_2_471609*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_471606И
'activity_regularization/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471616т
;activity_regularization/ActivityRegularizer/PartitionedCallPartitionedCall0activity_regularization/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_activity_regularization_activity_regularizer_471554С
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:Й
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskђ
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: џ
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ©
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_3_471638conv3d_3_471640*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_471637Ґ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_471655conv3d_4_471657*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_471654Ґ
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_471671conv3d_5_471673*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_471670Д
IdentityIdentity)conv3d_5/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: Ц
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
÷M
ф	
A__inference_model_layer_call_and_return_conditional_losses_472235

inputsC
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:	6
(conv3d_1_biasadd_readvariableop_resource:	E
'conv3d_2_conv3d_readvariableop_resource:	6
(conv3d_2_biasadd_readvariableop_resource:E
'conv3d_3_conv3d_readvariableop_resource:	6
(conv3d_3_biasadd_readvariableop_resource:	E
'conv3d_4_conv3d_readvariableop_resource:	6
(conv3d_4_biasadd_readvariableop_resource:E
'conv3d_5_conv3d_readvariableop_resource:6
(conv3d_5_biasadd_readvariableop_resource:
identity

identity_1ИҐconv3d/BiasAdd/ReadVariableOpҐconv3d/Conv3D/ReadVariableOpҐconv3d_1/BiasAdd/ReadVariableOpҐconv3d_1/Conv3D/ReadVariableOpҐconv3d_2/BiasAdd/ReadVariableOpҐconv3d_2/Conv3D/ReadVariableOpҐconv3d_3/BiasAdd/ReadVariableOpҐconv3d_3/Conv3D/ReadVariableOpҐconv3d_4/BiasAdd/ReadVariableOpҐconv3d_4/Conv3D/ReadVariableOpҐconv3d_5/BiasAdd/ReadVariableOpҐconv3d_5/Conv3D/ReadVariableOpО
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0ђ
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
А
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ц
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   Т
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0√
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
Д
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Ь
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	Т
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0≈
conv3d_2/Conv3DConv3Dconv3d_1/Relu:activations:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
Д
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   v
1activity_regularization/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    С
/activity_regularization/ActivityRegularizer/AbsAbsconv3d_2/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€   Р
3activity_regularization/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*)
value B"                 
/activity_regularization/ActivityRegularizer/SumSum3activity_regularization/ActivityRegularizer/Abs:y:0<activity_regularization/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: v
1activity_regularization/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЈQ8Ќ
/activity_regularization/ActivityRegularizer/mulMul:activity_regularization/ActivityRegularizer/mul/x:output:08activity_regularization/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
:  
/activity_regularization/ActivityRegularizer/addAddV2:activity_regularization/ActivityRegularizer/Const:output:03activity_regularization/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: |
1activity_regularization/ActivityRegularizer/ShapeShapeconv3d_2/Relu:activations:0*
T0*
_output_shapes
:Й
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskђ
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:  
3activity_regularization/ActivityRegularizer/truedivRealDiv3activity_regularization/ActivityRegularizer/add:z:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Т
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0≈
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
Д
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Ь
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	Т
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0≈
conv3d_4/Conv3DConv3Dconv3d_3/Relu:activations:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
Д
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   n
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   Т
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0≈
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
Д
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   t
IdentityIdentityconv3d_5/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ‘
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
…	
Г
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_472319

inputs
identity

identity_1Ќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471750∞
PartitionedCall_1PartitionedCallPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_activity_regularization_activity_regularizer_471554l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   S

Identity_1IdentityPartitionedCall_1:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   :[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
Њ
й
$__inference_signature_wrapper_472055
input_1%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCall…
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_471539{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€   
!
_user_specified_name	input_1
¶
Б
D__inference_conv3d_2_layer_call_and_return_conditional_losses_471606

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   	
 
_user_specified_nameinputs
¶
Б
D__inference_conv3d_3_layer_call_and_return_conditional_losses_471637

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
І
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471616

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:€€€€€€€€€   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   :[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
¶
Б
D__inference_conv3d_2_layer_call_and_return_conditional_losses_472295

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   	
 
_user_specified_nameinputs
І
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471750

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:€€€€€€€€€   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   :[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
…	
Г
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_472312

inputs
identity

identity_1Ќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471616∞
PartitionedCall_1PartitionedCallPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_activity_regularization_activity_regularizer_471554l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   S

Identity_1IdentityPartitionedCall_1:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   :[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
Б
Ґ
)__inference_conv3d_5_layer_call_fn_472368

inputs%
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_471670{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
г
л
&__inference_model_layer_call_fn_471706
input_1%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:	
	unknown_6:	'
	unknown_7:	
	unknown_8:'
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€   : *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_471678{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:€€€€€€€€€   
!
_user_specified_name	input_1
÷M
ф	
A__inference_model_layer_call_and_return_conditional_losses_472175

inputsC
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:	6
(conv3d_1_biasadd_readvariableop_resource:	E
'conv3d_2_conv3d_readvariableop_resource:	6
(conv3d_2_biasadd_readvariableop_resource:E
'conv3d_3_conv3d_readvariableop_resource:	6
(conv3d_3_biasadd_readvariableop_resource:	E
'conv3d_4_conv3d_readvariableop_resource:	6
(conv3d_4_biasadd_readvariableop_resource:E
'conv3d_5_conv3d_readvariableop_resource:6
(conv3d_5_biasadd_readvariableop_resource:
identity

identity_1ИҐconv3d/BiasAdd/ReadVariableOpҐconv3d/Conv3D/ReadVariableOpҐconv3d_1/BiasAdd/ReadVariableOpҐconv3d_1/Conv3D/ReadVariableOpҐconv3d_2/BiasAdd/ReadVariableOpҐconv3d_2/Conv3D/ReadVariableOpҐconv3d_3/BiasAdd/ReadVariableOpҐconv3d_3/Conv3D/ReadVariableOpҐconv3d_4/BiasAdd/ReadVariableOpҐconv3d_4/Conv3D/ReadVariableOpҐconv3d_5/BiasAdd/ReadVariableOpҐconv3d_5/Conv3D/ReadVariableOpО
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0ђ
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
А
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ц
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   Т
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0√
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
Д
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Ь
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	Т
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0≈
conv3d_2/Conv3DConv3Dconv3d_1/Relu:activations:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
Д
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   v
1activity_regularization/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    С
/activity_regularization/ActivityRegularizer/AbsAbsconv3d_2/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€   Р
3activity_regularization/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*)
value B"                 
/activity_regularization/ActivityRegularizer/SumSum3activity_regularization/ActivityRegularizer/Abs:y:0<activity_regularization/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: v
1activity_regularization/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЈQ8Ќ
/activity_regularization/ActivityRegularizer/mulMul:activity_regularization/ActivityRegularizer/mul/x:output:08activity_regularization/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
:  
/activity_regularization/ActivityRegularizer/addAddV2:activity_regularization/ActivityRegularizer/Const:output:03activity_regularization/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: |
1activity_regularization/ActivityRegularizer/ShapeShapeconv3d_2/Relu:activations:0*
T0*
_output_shapes
:Й
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:≠
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskђ
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
:  
3activity_regularization/ActivityRegularizer/truedivRealDiv3activity_regularization/ActivityRegularizer/add:z:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Т
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0≈
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	*
paddingSAME*
strides	
Д
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Ь
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   	n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   	Т
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0≈
conv3d_4/Conv3DConv3Dconv3d_3/Relu:activations:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
Д
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   n
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   Т
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0≈
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
Д
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   t
IdentityIdentityconv3d_5/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ‘
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:€€€€€€€€€   : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
о
T
8__inference_activity_regularization_layer_call_fn_472300

inputs
identityЌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_471616l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   :[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs
Б
Ґ
)__inference_conv3d_4_layer_call_fn_472348

inputs%
unknown:	
	unknown_0:
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_471654{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   	: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€   	
 
_user_specified_nameinputs
¶
Б
D__inference_conv3d_4_layer_call_and_return_conditional_losses_471654

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   	
 
_user_specified_nameinputs
§
€
B__inference_conv3d_layer_call_and_return_conditional_losses_471572

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0Ю
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€   
 
_user_specified_nameinputs"ВL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*√
serving_defaultѓ
G
input_1<
serving_default_input_1:0€€€€€€€€€   H
conv3d_5<
StatefulPartitionedCall:0€€€€€€€€€   tensorflow/serving/predict:≥Ъ
к
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
Б__call__
+В&call_and_return_all_conditional_losses
Г_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
љ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"
_tf_keras_layer
І
!	variables
"trainable_variables
#regularization_losses
$	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
ђ
7iter

8beta_1

9beta_2
	:decay
;learning_ratemimjmkmlmmmn%mo&mp+mq,mr1ms2mtvuvvvwvxvyvz%v{&v|+v},v~1v2vА"
	optimizer
v
0
1
2
3
4
5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
v
0
1
2
3
4
5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
 "
trackable_list_wrapper
ќ
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics

	variables
trainable_variables
regularization_losses
Б__call__
Г_default_save_signature
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
-
Тserving_default"
signature_map
+:)2conv3d/kernel
:2conv3d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_1/kernel
:	2conv3d_1/bias
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
∞
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_2/kernel
:2conv3d_2/bias
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
∞
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ќ
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
!	variables
"trainable_variables
#regularization_losses
К__call__
Уactivity_regularizer_fn
+Л&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_3/kernel
:	2conv3d_3/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
'	variables
(trainable_variables
)regularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_4/kernel
:2conv3d_4/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
-	variables
.trainable_variables
/regularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_5/kernel
:2conv3d_5/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
3	variables
4trainable_variables
5regularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
d0"
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
N
	etotal
	fcount
g	variables
h	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
e0
f1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
0:.2Adam/conv3d/kernel/m
:2Adam/conv3d/bias/m
2:0	2Adam/conv3d_1/kernel/m
 :	2Adam/conv3d_1/bias/m
2:0	2Adam/conv3d_2/kernel/m
 :2Adam/conv3d_2/bias/m
2:0	2Adam/conv3d_3/kernel/m
 :	2Adam/conv3d_3/bias/m
2:0	2Adam/conv3d_4/kernel/m
 :2Adam/conv3d_4/bias/m
2:02Adam/conv3d_5/kernel/m
 :2Adam/conv3d_5/bias/m
0:.2Adam/conv3d/kernel/v
:2Adam/conv3d/bias/v
2:0	2Adam/conv3d_1/kernel/v
 :	2Adam/conv3d_1/bias/v
2:0	2Adam/conv3d_2/kernel/v
 :2Adam/conv3d_2/bias/v
2:0	2Adam/conv3d_3/kernel/v
 :	2Adam/conv3d_3/bias/v
2:0	2Adam/conv3d_4/kernel/v
 :2Adam/conv3d_4/bias/v
2:02Adam/conv3d_5/kernel/v
 :2Adam/conv3d_5/bias/v
ж2г
&__inference_model_layer_call_fn_471706
&__inference_model_layer_call_fn_472085
&__inference_model_layer_call_fn_472115
&__inference_model_layer_call_fn_471930ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
“2ѕ
A__inference_model_layer_call_and_return_conditional_losses_472175
A__inference_model_layer_call_and_return_conditional_losses_472235
A__inference_model_layer_call_and_return_conditional_losses_471974
A__inference_model_layer_call_and_return_conditional_losses_472018ј
Ј≤≥
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
kwonlydefaults™ 
annotations™ *
 
ћB…
!__inference__wrapped_model_471539input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv3d_layer_call_fn_472244Ґ
Щ≤Х
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
annotations™ *
 
м2й
B__inference_conv3d_layer_call_and_return_conditional_losses_472255Ґ
Щ≤Х
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
annotations™ *
 
”2–
)__inference_conv3d_1_layer_call_fn_472264Ґ
Щ≤Х
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
annotations™ *
 
о2л
D__inference_conv3d_1_layer_call_and_return_conditional_losses_472275Ґ
Щ≤Х
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
annotations™ *
 
”2–
)__inference_conv3d_2_layer_call_fn_472284Ґ
Щ≤Х
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
annotations™ *
 
о2л
D__inference_conv3d_2_layer_call_and_return_conditional_losses_472295Ґ
Щ≤Х
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
annotations™ *
 
ƒ2Ѕ
8__inference_activity_regularization_layer_call_fn_472300
8__inference_activity_regularization_layer_call_fn_472305 
Ѕ≤љ
FullArgSpec
argsЪ
jself
jinputs
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
В2€
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_472312
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_472319 
Ѕ≤љ
FullArgSpec
argsЪ
jself
jinputs
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
”2–
)__inference_conv3d_3_layer_call_fn_472328Ґ
Щ≤Х
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
annotations™ *
 
о2л
D__inference_conv3d_3_layer_call_and_return_conditional_losses_472339Ґ
Щ≤Х
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
annotations™ *
 
”2–
)__inference_conv3d_4_layer_call_fn_472348Ґ
Щ≤Х
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
annotations™ *
 
о2л
D__inference_conv3d_4_layer_call_and_return_conditional_losses_472359Ґ
Щ≤Х
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
annotations™ *
 
”2–
)__inference_conv3d_5_layer_call_fn_472368Ґ
Щ≤Х
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
annotations™ *
 
о2л
D__inference_conv3d_5_layer_call_and_return_conditional_losses_472378Ґ
Щ≤Х
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
annotations™ *
 
ЋB»
$__inference_signature_wrapper_472055input_1"Ф
Н≤Й
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
annotations™ *
 
р2н
?__inference_activity_regularization_activity_regularizer_471554©
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ
	К
ъ2ч
S__inference_activity_regularization_layer_call_and_return_conditional_losses_472382
S__inference_activity_regularization_layer_call_and_return_conditional_losses_472386 
Ѕ≤љ
FullArgSpec
argsЪ
jself
jinputs
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 ≥
!__inference__wrapped_model_471539Н%&+,12<Ґ9
2Ґ/
-К*
input_1€€€€€€€€€   
™ "?™<
:
conv3d_5.К+
conv3d_5€€€€€€€€€   i
?__inference_activity_regularization_activity_regularizer_471554&Ґ
Ґ
К	
x
™ "К к
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_472312ОKҐH
1Ґ.
,К)
inputs€€€€€€€€€   
™

trainingp "?Ґ<
'К$
0€€€€€€€€€   
Ъ
К	
1/0 к
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_472319ОKҐH
1Ґ.
,К)
inputs€€€€€€€€€   
™

trainingp"?Ґ<
'К$
0€€€€€€€€€   
Ъ
К	
1/0 Ў
S__inference_activity_regularization_layer_call_and_return_conditional_losses_472382АKҐH
1Ґ.
,К)
inputs€€€€€€€€€   
™

trainingp "1Ґ.
'К$
0€€€€€€€€€   
Ъ Ў
S__inference_activity_regularization_layer_call_and_return_conditional_losses_472386АKҐH
1Ґ.
,К)
inputs€€€€€€€€€   
™

trainingp"1Ґ.
'К$
0€€€€€€€€€   
Ъ ѓ
8__inference_activity_regularization_layer_call_fn_472300sKҐH
1Ґ.
,К)
inputs€€€€€€€€€   
™

trainingp "$К!€€€€€€€€€   ѓ
8__inference_activity_regularization_layer_call_fn_472305sKҐH
1Ґ.
,К)
inputs€€€€€€€€€   
™

trainingp"$К!€€€€€€€€€   Љ
D__inference_conv3d_1_layer_call_and_return_conditional_losses_472275t;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   
™ "1Ґ.
'К$
0€€€€€€€€€   	
Ъ Ф
)__inference_conv3d_1_layer_call_fn_472264g;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   
™ "$К!€€€€€€€€€   	Љ
D__inference_conv3d_2_layer_call_and_return_conditional_losses_472295t;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   	
™ "1Ґ.
'К$
0€€€€€€€€€   
Ъ Ф
)__inference_conv3d_2_layer_call_fn_472284g;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   	
™ "$К!€€€€€€€€€   Љ
D__inference_conv3d_3_layer_call_and_return_conditional_losses_472339t%&;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   
™ "1Ґ.
'К$
0€€€€€€€€€   	
Ъ Ф
)__inference_conv3d_3_layer_call_fn_472328g%&;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   
™ "$К!€€€€€€€€€   	Љ
D__inference_conv3d_4_layer_call_and_return_conditional_losses_472359t+,;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   	
™ "1Ґ.
'К$
0€€€€€€€€€   
Ъ Ф
)__inference_conv3d_4_layer_call_fn_472348g+,;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   	
™ "$К!€€€€€€€€€   Љ
D__inference_conv3d_5_layer_call_and_return_conditional_losses_472378t12;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   
™ "1Ґ.
'К$
0€€€€€€€€€   
Ъ Ф
)__inference_conv3d_5_layer_call_fn_472368g12;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   
™ "$К!€€€€€€€€€   Ї
B__inference_conv3d_layer_call_and_return_conditional_losses_472255t;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   
™ "1Ґ.
'К$
0€€€€€€€€€   
Ъ Т
'__inference_conv3d_layer_call_fn_472244g;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€   
™ "$К!€€€€€€€€€   џ
A__inference_model_layer_call_and_return_conditional_losses_471974Х%&+,12DҐA
:Ґ7
-К*
input_1€€€€€€€€€   
p 

 
™ "?Ґ<
'К$
0€€€€€€€€€   
Ъ
К	
1/0 џ
A__inference_model_layer_call_and_return_conditional_losses_472018Х%&+,12DҐA
:Ґ7
-К*
input_1€€€€€€€€€   
p

 
™ "?Ґ<
'К$
0€€€€€€€€€   
Ъ
К	
1/0 Џ
A__inference_model_layer_call_and_return_conditional_losses_472175Ф%&+,12CҐ@
9Ґ6
,К)
inputs€€€€€€€€€   
p 

 
™ "?Ґ<
'К$
0€€€€€€€€€   
Ъ
К	
1/0 Џ
A__inference_model_layer_call_and_return_conditional_losses_472235Ф%&+,12CҐ@
9Ґ6
,К)
inputs€€€€€€€€€   
p

 
™ "?Ґ<
'К$
0€€€€€€€€€   
Ъ
К	
1/0 §
&__inference_model_layer_call_fn_471706z%&+,12DҐA
:Ґ7
-К*
input_1€€€€€€€€€   
p 

 
™ "$К!€€€€€€€€€   §
&__inference_model_layer_call_fn_471930z%&+,12DҐA
:Ґ7
-К*
input_1€€€€€€€€€   
p

 
™ "$К!€€€€€€€€€   £
&__inference_model_layer_call_fn_472085y%&+,12CҐ@
9Ґ6
,К)
inputs€€€€€€€€€   
p 

 
™ "$К!€€€€€€€€€   £
&__inference_model_layer_call_fn_472115y%&+,12CҐ@
9Ґ6
,К)
inputs€€€€€€€€€   
p

 
™ "$К!€€€€€€€€€   Ѕ
$__inference_signature_wrapper_472055Ш%&+,12GҐD
Ґ 
=™:
8
input_1-К*
input_1€€€€€€€€€   "?™<
:
conv3d_5.К+
conv3d_5€€€€€€€€€   