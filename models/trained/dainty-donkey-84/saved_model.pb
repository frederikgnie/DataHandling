��
��
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
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
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
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
�
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
�
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
�
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
�
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:*
dtype0
�
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_4/kernel

#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel**
_output_shapes
:*
dtype0
r
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:*
dtype0
�
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
:*
dtype0
r
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
:*
dtype0
�
conv3d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_6/kernel

#conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv3d_6/kernel**
_output_shapes
:*
dtype0
r
conv3d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_6/bias
k
!conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv3d_6/bias*
_output_shapes
:*
dtype0
�
conv3d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_7/kernel

#conv3d_7/kernel/Read/ReadVariableOpReadVariableOpconv3d_7/kernel**
_output_shapes
:*
dtype0
r
conv3d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_7/bias
k
!conv3d_7/bias/Read/ReadVariableOpReadVariableOpconv3d_7/bias*
_output_shapes
:*
dtype0
�
conv3d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_8/kernel

#conv3d_8/kernel/Read/ReadVariableOpReadVariableOpconv3d_8/kernel**
_output_shapes
:*
dtype0
r
conv3d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_8/bias
k
!conv3d_8/bias/Read/ReadVariableOpReadVariableOpconv3d_8/bias*
_output_shapes
:*
dtype0
�
conv3d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_9/kernel

#conv3d_9/kernel/Read/ReadVariableOpReadVariableOpconv3d_9/kernel**
_output_shapes
:*
dtype0
r
conv3d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_9/bias
k
!conv3d_9/bias/Read/ReadVariableOpReadVariableOpconv3d_9/bias*
_output_shapes
:*
dtype0
�
conv3d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_10/kernel
�
$conv3d_10/kernel/Read/ReadVariableOpReadVariableOpconv3d_10/kernel**
_output_shapes
:*
dtype0
t
conv3d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_10/bias
m
"conv3d_10/bias/Read/ReadVariableOpReadVariableOpconv3d_10/bias*
_output_shapes
:*
dtype0
�
conv3d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_11/kernel
�
$conv3d_11/kernel/Read/ReadVariableOpReadVariableOpconv3d_11/kernel**
_output_shapes
:*
dtype0
t
conv3d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_11/bias
m
"conv3d_11/bias/Read/ReadVariableOpReadVariableOpconv3d_11/bias*
_output_shapes
:*
dtype0
�
conv3d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_12/kernel
�
$conv3d_12/kernel/Read/ReadVariableOpReadVariableOpconv3d_12/kernel**
_output_shapes
:*
dtype0
t
conv3d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_12/bias
m
"conv3d_12/bias/Read/ReadVariableOpReadVariableOpconv3d_12/bias*
_output_shapes
:*
dtype0
�
conv3d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_13/kernel
�
$conv3d_13/kernel/Read/ReadVariableOpReadVariableOpconv3d_13/kernel**
_output_shapes
:*
dtype0
t
conv3d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_13/bias
m
"conv3d_13/bias/Read/ReadVariableOpReadVariableOpconv3d_13/bias*
_output_shapes
:*
dtype0
�
conv3d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_14/kernel
�
$conv3d_14/kernel/Read/ReadVariableOpReadVariableOpconv3d_14/kernel**
_output_shapes
:*
dtype0
t
conv3d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_14/bias
m
"conv3d_14/bias/Read/ReadVariableOpReadVariableOpconv3d_14/bias*
_output_shapes
:*
dtype0
�
conv3d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_nameconv3d_15/kernel
�
$conv3d_15/kernel/Read/ReadVariableOpReadVariableOpconv3d_15/kernel**
_output_shapes
:	*
dtype0
t
conv3d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv3d_15/bias
m
"conv3d_15/bias/Read/ReadVariableOpReadVariableOpconv3d_15/bias*
_output_shapes
:	*
dtype0
�
conv3d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_nameconv3d_16/kernel
�
$conv3d_16/kernel/Read/ReadVariableOpReadVariableOpconv3d_16/kernel**
_output_shapes
:	*
dtype0
t
conv3d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_16/bias
m
"conv3d_16/bias/Read/ReadVariableOpReadVariableOpconv3d_16/bias*
_output_shapes
:*
dtype0
�
conv3d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv3d_17/kernel
�
$conv3d_17/kernel/Read/ReadVariableOpReadVariableOpconv3d_17/kernel**
_output_shapes
:*
dtype0
t
conv3d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_17/bias
m
"conv3d_17/bias/Read/ReadVariableOpReadVariableOpconv3d_17/bias*
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
�
Adam/conv3d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/m
�
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
�
Adam/conv3d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_1/kernel/m
�
*Adam/conv3d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/m**
_output_shapes
:	*
dtype0
�
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
�
Adam/conv3d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_2/kernel/m
�
*Adam/conv3d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/m**
_output_shapes
:	*
dtype0
�
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
�
Adam/conv3d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_3/kernel/m
�
*Adam/conv3d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_3/bias/m
y
(Adam/conv3d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_4/kernel/m
�
*Adam/conv3d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_4/bias/m
y
(Adam/conv3d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/m
�
*Adam/conv3d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_5/bias/m
y
(Adam/conv3d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_6/kernel/m
�
*Adam/conv3d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_6/bias/m
y
(Adam/conv3d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_7/kernel/m
�
*Adam/conv3d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_7/bias/m
y
(Adam/conv3d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_8/kernel/m
�
*Adam/conv3d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_8/bias/m
y
(Adam/conv3d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_9/kernel/m
�
*Adam/conv3d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_9/bias/m
y
(Adam/conv3d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_10/kernel/m
�
+Adam/conv3d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_10/bias/m
{
)Adam/conv3d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_11/kernel/m
�
+Adam/conv3d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_11/bias/m
{
)Adam/conv3d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_12/kernel/m
�
+Adam/conv3d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_12/bias/m
{
)Adam/conv3d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_13/kernel/m
�
+Adam/conv3d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_13/bias/m
{
)Adam/conv3d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_14/kernel/m
�
+Adam/conv3d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_14/bias/m
{
)Adam/conv3d_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/conv3d_15/kernel/m
�
+Adam/conv3d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/kernel/m**
_output_shapes
:	*
dtype0
�
Adam/conv3d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/conv3d_15/bias/m
{
)Adam/conv3d_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/bias/m*
_output_shapes
:	*
dtype0
�
Adam/conv3d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/conv3d_16/kernel/m
�
+Adam/conv3d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/kernel/m**
_output_shapes
:	*
dtype0
�
Adam/conv3d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_16/bias/m
{
)Adam/conv3d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_17/kernel/m
�
+Adam/conv3d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/kernel/m**
_output_shapes
:*
dtype0
�
Adam/conv3d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_17/bias/m
{
)Adam/conv3d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv3d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d/kernel/v
�
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
�
Adam/conv3d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_1/kernel/v
�
*Adam/conv3d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/v**
_output_shapes
:	*
dtype0
�
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
�
Adam/conv3d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv3d_2/kernel/v
�
*Adam/conv3d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/v**
_output_shapes
:	*
dtype0
�
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
�
Adam/conv3d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_3/kernel/v
�
*Adam/conv3d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_3/bias/v
y
(Adam/conv3d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_4/kernel/v
�
*Adam/conv3d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_4/bias/v
y
(Adam/conv3d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/v
�
*Adam/conv3d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_5/bias/v
y
(Adam/conv3d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_6/kernel/v
�
*Adam/conv3d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_6/bias/v
y
(Adam/conv3d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_7/kernel/v
�
*Adam/conv3d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_7/bias/v
y
(Adam/conv3d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_8/kernel/v
�
*Adam/conv3d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_8/bias/v
y
(Adam/conv3d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_8/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_9/kernel/v
�
*Adam/conv3d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_9/bias/v
y
(Adam/conv3d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_9/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_10/kernel/v
�
+Adam/conv3d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_10/bias/v
{
)Adam/conv3d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_10/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_11/kernel/v
�
+Adam/conv3d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_11/bias/v
{
)Adam/conv3d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_11/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_12/kernel/v
�
+Adam/conv3d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_12/bias/v
{
)Adam/conv3d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_12/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_13/kernel/v
�
+Adam/conv3d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_13/bias/v
{
)Adam/conv3d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_13/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_14/kernel/v
�
+Adam/conv3d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_14/bias/v
{
)Adam/conv3d_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_14/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/conv3d_15/kernel/v
�
+Adam/conv3d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/kernel/v**
_output_shapes
:	*
dtype0
�
Adam/conv3d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/conv3d_15/bias/v
{
)Adam/conv3d_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_15/bias/v*
_output_shapes
:	*
dtype0
�
Adam/conv3d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/conv3d_16/kernel/v
�
+Adam/conv3d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/kernel/v**
_output_shapes
:	*
dtype0
�
Adam/conv3d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_16/bias/v
{
)Adam/conv3d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_16/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv3d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv3d_17/kernel/v
�
+Adam/conv3d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/kernel/v**
_output_shapes
:*
dtype0
�
Adam/conv3d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv3d_17/bias/v
{
)Adam/conv3d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_17/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ӫ
valueȫBī B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer_with_weights-15
layer-17
layer_with_weights-16
layer-18
layer_with_weights-17
layer-19
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
h

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
h

Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
R
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
h

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
h

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
h

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
h

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
h

ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
m

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�!m�"m�'m�(m�-m�.m�3m�4m�9m�:m�?m�@m�Em�Fm�Km�Lm�Um�Vm�[m�\m�am�bm�gm�hm�mm�nm�sm�tm�ym�zm�m�	�m�	�m�	�m�v�v�!v�"v�'v�(v�-v�.v�3v�4v�9v�:v�?v�@v�Ev�Fv�Kv�Lv�Uv�Vv�[v�\v�av�bv�gv�hv�mv�nv�sv�tv�yv�zv�v�	�v�	�v�	�v�
�
0
1
!2
"3
'4
(5
-6
.7
38
49
910
:11
?12
@13
E14
F15
K16
L17
U18
V19
[20
\21
a22
b23
g24
h25
m26
n27
s28
t29
y30
z31
32
�33
�34
�35
�
0
1
!2
"3
'4
(5
-6
.7
38
49
910
:11
?12
@13
E14
F15
K16
L17
U18
V19
[20
\21
a22
b23
g24
h25
m26
n27
s28
t29
y30
z31
32
�33
�34
�35
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 
YW
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
[Y
VARIABLE_VALUEconv3d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
[Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
[Y
VARIABLE_VALUEconv3d_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
[Y
VARIABLE_VALUEconv3d_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

E0
F1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
[Y
VARIABLE_VALUEconv3d_8/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_8/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

K0
L1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
[Y
VARIABLE_VALUEconv3d_9/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_9/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
][
VARIABLE_VALUEconv3d_10/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_10/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

[0
\1

[0
\1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
][
VARIABLE_VALUEconv3d_11/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_11/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

a0
b1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
][
VARIABLE_VALUEconv3d_12/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_12/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
][
VARIABLE_VALUEconv3d_13/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_13/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

m0
n1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
][
VARIABLE_VALUEconv3d_14/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_14/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
][
VARIABLE_VALUEconv3d_15/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_15/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

y0
z1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
][
VARIABLE_VALUEconv3d_16/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_16/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

0
�1

0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
][
VARIABLE_VALUEconv3d_17/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv3d_17/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
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
�
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
11
12
13
14
15
16
17
18
19

�0
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
8

�total

�count
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
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
~|
VARIABLE_VALUEAdam/conv3d_6/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_6/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_7/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_7/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_8/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_8/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_9/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_9/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_10/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_10/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_11/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_11/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_12/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_12/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_13/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_13/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_14/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_14/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_15/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_15/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_16/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_16/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_17/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_17/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
~|
VARIABLE_VALUEAdam/conv3d_6/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_6/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_7/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_7/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_8/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_8/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_9/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_9/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_10/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_10/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_11/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_11/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_12/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_12/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_13/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_13/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_14/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_14/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_15/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_15/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_16/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_16/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdam/conv3d_17/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv3d_17/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*3
_output_shapes!
:���������   *
dtype0*(
shape:���������   
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_6/kernelconv3d_6/biasconv3d_7/kernelconv3d_7/biasconv3d_8/kernelconv3d_8/biasconv3d_9/kernelconv3d_9/biasconv3d_10/kernelconv3d_10/biasconv3d_11/kernelconv3d_11/biasconv3d_12/kernelconv3d_12/biasconv3d_13/kernelconv3d_13/biasconv3d_14/kernelconv3d_14/biasconv3d_15/kernelconv3d_15/biasconv3d_16/kernelconv3d_16/biasconv3d_17/kernelconv3d_17/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_386143
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp#conv3d_6/kernel/Read/ReadVariableOp!conv3d_6/bias/Read/ReadVariableOp#conv3d_7/kernel/Read/ReadVariableOp!conv3d_7/bias/Read/ReadVariableOp#conv3d_8/kernel/Read/ReadVariableOp!conv3d_8/bias/Read/ReadVariableOp#conv3d_9/kernel/Read/ReadVariableOp!conv3d_9/bias/Read/ReadVariableOp$conv3d_10/kernel/Read/ReadVariableOp"conv3d_10/bias/Read/ReadVariableOp$conv3d_11/kernel/Read/ReadVariableOp"conv3d_11/bias/Read/ReadVariableOp$conv3d_12/kernel/Read/ReadVariableOp"conv3d_12/bias/Read/ReadVariableOp$conv3d_13/kernel/Read/ReadVariableOp"conv3d_13/bias/Read/ReadVariableOp$conv3d_14/kernel/Read/ReadVariableOp"conv3d_14/bias/Read/ReadVariableOp$conv3d_15/kernel/Read/ReadVariableOp"conv3d_15/bias/Read/ReadVariableOp$conv3d_16/kernel/Read/ReadVariableOp"conv3d_16/bias/Read/ReadVariableOp$conv3d_17/kernel/Read/ReadVariableOp"conv3d_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv3d/kernel/m/Read/ReadVariableOp&Adam/conv3d/bias/m/Read/ReadVariableOp*Adam/conv3d_1/kernel/m/Read/ReadVariableOp(Adam/conv3d_1/bias/m/Read/ReadVariableOp*Adam/conv3d_2/kernel/m/Read/ReadVariableOp(Adam/conv3d_2/bias/m/Read/ReadVariableOp*Adam/conv3d_3/kernel/m/Read/ReadVariableOp(Adam/conv3d_3/bias/m/Read/ReadVariableOp*Adam/conv3d_4/kernel/m/Read/ReadVariableOp(Adam/conv3d_4/bias/m/Read/ReadVariableOp*Adam/conv3d_5/kernel/m/Read/ReadVariableOp(Adam/conv3d_5/bias/m/Read/ReadVariableOp*Adam/conv3d_6/kernel/m/Read/ReadVariableOp(Adam/conv3d_6/bias/m/Read/ReadVariableOp*Adam/conv3d_7/kernel/m/Read/ReadVariableOp(Adam/conv3d_7/bias/m/Read/ReadVariableOp*Adam/conv3d_8/kernel/m/Read/ReadVariableOp(Adam/conv3d_8/bias/m/Read/ReadVariableOp*Adam/conv3d_9/kernel/m/Read/ReadVariableOp(Adam/conv3d_9/bias/m/Read/ReadVariableOp+Adam/conv3d_10/kernel/m/Read/ReadVariableOp)Adam/conv3d_10/bias/m/Read/ReadVariableOp+Adam/conv3d_11/kernel/m/Read/ReadVariableOp)Adam/conv3d_11/bias/m/Read/ReadVariableOp+Adam/conv3d_12/kernel/m/Read/ReadVariableOp)Adam/conv3d_12/bias/m/Read/ReadVariableOp+Adam/conv3d_13/kernel/m/Read/ReadVariableOp)Adam/conv3d_13/bias/m/Read/ReadVariableOp+Adam/conv3d_14/kernel/m/Read/ReadVariableOp)Adam/conv3d_14/bias/m/Read/ReadVariableOp+Adam/conv3d_15/kernel/m/Read/ReadVariableOp)Adam/conv3d_15/bias/m/Read/ReadVariableOp+Adam/conv3d_16/kernel/m/Read/ReadVariableOp)Adam/conv3d_16/bias/m/Read/ReadVariableOp+Adam/conv3d_17/kernel/m/Read/ReadVariableOp)Adam/conv3d_17/bias/m/Read/ReadVariableOp(Adam/conv3d/kernel/v/Read/ReadVariableOp&Adam/conv3d/bias/v/Read/ReadVariableOp*Adam/conv3d_1/kernel/v/Read/ReadVariableOp(Adam/conv3d_1/bias/v/Read/ReadVariableOp*Adam/conv3d_2/kernel/v/Read/ReadVariableOp(Adam/conv3d_2/bias/v/Read/ReadVariableOp*Adam/conv3d_3/kernel/v/Read/ReadVariableOp(Adam/conv3d_3/bias/v/Read/ReadVariableOp*Adam/conv3d_4/kernel/v/Read/ReadVariableOp(Adam/conv3d_4/bias/v/Read/ReadVariableOp*Adam/conv3d_5/kernel/v/Read/ReadVariableOp(Adam/conv3d_5/bias/v/Read/ReadVariableOp*Adam/conv3d_6/kernel/v/Read/ReadVariableOp(Adam/conv3d_6/bias/v/Read/ReadVariableOp*Adam/conv3d_7/kernel/v/Read/ReadVariableOp(Adam/conv3d_7/bias/v/Read/ReadVariableOp*Adam/conv3d_8/kernel/v/Read/ReadVariableOp(Adam/conv3d_8/bias/v/Read/ReadVariableOp*Adam/conv3d_9/kernel/v/Read/ReadVariableOp(Adam/conv3d_9/bias/v/Read/ReadVariableOp+Adam/conv3d_10/kernel/v/Read/ReadVariableOp)Adam/conv3d_10/bias/v/Read/ReadVariableOp+Adam/conv3d_11/kernel/v/Read/ReadVariableOp)Adam/conv3d_11/bias/v/Read/ReadVariableOp+Adam/conv3d_12/kernel/v/Read/ReadVariableOp)Adam/conv3d_12/bias/v/Read/ReadVariableOp+Adam/conv3d_13/kernel/v/Read/ReadVariableOp)Adam/conv3d_13/bias/v/Read/ReadVariableOp+Adam/conv3d_14/kernel/v/Read/ReadVariableOp)Adam/conv3d_14/bias/v/Read/ReadVariableOp+Adam/conv3d_15/kernel/v/Read/ReadVariableOp)Adam/conv3d_15/bias/v/Read/ReadVariableOp+Adam/conv3d_16/kernel/v/Read/ReadVariableOp)Adam/conv3d_16/bias/v/Read/ReadVariableOp+Adam/conv3d_17/kernel/v/Read/ReadVariableOp)Adam/conv3d_17/bias/v/Read/ReadVariableOpConst*�
Tiny
w2u	*
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
GPU2*0J 8� *(
f#R!
__inference__traced_save_387346
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_6/kernelconv3d_6/biasconv3d_7/kernelconv3d_7/biasconv3d_8/kernelconv3d_8/biasconv3d_9/kernelconv3d_9/biasconv3d_10/kernelconv3d_10/biasconv3d_11/kernelconv3d_11/biasconv3d_12/kernelconv3d_12/biasconv3d_13/kernelconv3d_13/biasconv3d_14/kernelconv3d_14/biasconv3d_15/kernelconv3d_15/biasconv3d_16/kernelconv3d_16/biasconv3d_17/kernelconv3d_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv3d/kernel/mAdam/conv3d/bias/mAdam/conv3d_1/kernel/mAdam/conv3d_1/bias/mAdam/conv3d_2/kernel/mAdam/conv3d_2/bias/mAdam/conv3d_3/kernel/mAdam/conv3d_3/bias/mAdam/conv3d_4/kernel/mAdam/conv3d_4/bias/mAdam/conv3d_5/kernel/mAdam/conv3d_5/bias/mAdam/conv3d_6/kernel/mAdam/conv3d_6/bias/mAdam/conv3d_7/kernel/mAdam/conv3d_7/bias/mAdam/conv3d_8/kernel/mAdam/conv3d_8/bias/mAdam/conv3d_9/kernel/mAdam/conv3d_9/bias/mAdam/conv3d_10/kernel/mAdam/conv3d_10/bias/mAdam/conv3d_11/kernel/mAdam/conv3d_11/bias/mAdam/conv3d_12/kernel/mAdam/conv3d_12/bias/mAdam/conv3d_13/kernel/mAdam/conv3d_13/bias/mAdam/conv3d_14/kernel/mAdam/conv3d_14/bias/mAdam/conv3d_15/kernel/mAdam/conv3d_15/bias/mAdam/conv3d_16/kernel/mAdam/conv3d_16/bias/mAdam/conv3d_17/kernel/mAdam/conv3d_17/bias/mAdam/conv3d/kernel/vAdam/conv3d/bias/vAdam/conv3d_1/kernel/vAdam/conv3d_1/bias/vAdam/conv3d_2/kernel/vAdam/conv3d_2/bias/vAdam/conv3d_3/kernel/vAdam/conv3d_3/bias/vAdam/conv3d_4/kernel/vAdam/conv3d_4/bias/vAdam/conv3d_5/kernel/vAdam/conv3d_5/bias/vAdam/conv3d_6/kernel/vAdam/conv3d_6/bias/vAdam/conv3d_7/kernel/vAdam/conv3d_7/bias/vAdam/conv3d_8/kernel/vAdam/conv3d_8/bias/vAdam/conv3d_9/kernel/vAdam/conv3d_9/bias/vAdam/conv3d_10/kernel/vAdam/conv3d_10/bias/vAdam/conv3d_11/kernel/vAdam/conv3d_11/bias/vAdam/conv3d_12/kernel/vAdam/conv3d_12/bias/vAdam/conv3d_13/kernel/vAdam/conv3d_13/bias/vAdam/conv3d_14/kernel/vAdam/conv3d_14/bias/vAdam/conv3d_15/kernel/vAdam/conv3d_15/bias/vAdam/conv3d_16/kernel/vAdam/conv3d_16/bias/vAdam/conv3d_17/kernel/vAdam/conv3d_17/bias/v*
Tinx
v2t*
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
GPU2*0J 8� *+
f&R$
"__inference__traced_restore_387701��
�
�
D__inference_conv3d_1_layer_call_and_return_conditional_losses_386627

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_7_layer_call_and_return_conditional_losses_385035

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_4_layer_call_fn_386676

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_384984{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
B__inference_conv3d_layer_call_and_return_conditional_losses_384916

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
E__inference_conv3d_11_layer_call_and_return_conditional_losses_386851

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
*__inference_conv3d_13_layer_call_fn_386880

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_385151{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�

&__inference_model_layer_call_fn_385302
input_1%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:	

unknown_30:	(

unknown_31:	

unknown_32:(

unknown_33:

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:���������   : *F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_385226{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
E__inference_conv3d_11_layer_call_and_return_conditional_losses_385117

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_386978

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
��
�
A__inference_model_layer_call_and_return_conditional_losses_386443

inputsC
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:	6
(conv3d_1_biasadd_readvariableop_resource:	E
'conv3d_2_conv3d_readvariableop_resource:	6
(conv3d_2_biasadd_readvariableop_resource:E
'conv3d_3_conv3d_readvariableop_resource:6
(conv3d_3_biasadd_readvariableop_resource:E
'conv3d_4_conv3d_readvariableop_resource:6
(conv3d_4_biasadd_readvariableop_resource:E
'conv3d_5_conv3d_readvariableop_resource:6
(conv3d_5_biasadd_readvariableop_resource:E
'conv3d_6_conv3d_readvariableop_resource:6
(conv3d_6_biasadd_readvariableop_resource:E
'conv3d_7_conv3d_readvariableop_resource:6
(conv3d_7_biasadd_readvariableop_resource:E
'conv3d_8_conv3d_readvariableop_resource:6
(conv3d_8_biasadd_readvariableop_resource:E
'conv3d_9_conv3d_readvariableop_resource:6
(conv3d_9_biasadd_readvariableop_resource:F
(conv3d_10_conv3d_readvariableop_resource:7
)conv3d_10_biasadd_readvariableop_resource:F
(conv3d_11_conv3d_readvariableop_resource:7
)conv3d_11_biasadd_readvariableop_resource:F
(conv3d_12_conv3d_readvariableop_resource:7
)conv3d_12_biasadd_readvariableop_resource:F
(conv3d_13_conv3d_readvariableop_resource:7
)conv3d_13_biasadd_readvariableop_resource:F
(conv3d_14_conv3d_readvariableop_resource:7
)conv3d_14_biasadd_readvariableop_resource:F
(conv3d_15_conv3d_readvariableop_resource:	7
)conv3d_15_biasadd_readvariableop_resource:	F
(conv3d_16_conv3d_readvariableop_resource:	7
)conv3d_16_biasadd_readvariableop_resource:F
(conv3d_17_conv3d_readvariableop_resource:7
)conv3d_17_biasadd_readvariableop_resource:
identity

identity_1��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp� conv3d_10/BiasAdd/ReadVariableOp�conv3d_10/Conv3D/ReadVariableOp� conv3d_11/BiasAdd/ReadVariableOp�conv3d_11/Conv3D/ReadVariableOp� conv3d_12/BiasAdd/ReadVariableOp�conv3d_12/Conv3D/ReadVariableOp� conv3d_13/BiasAdd/ReadVariableOp�conv3d_13/Conv3D/ReadVariableOp� conv3d_14/BiasAdd/ReadVariableOp�conv3d_14/Conv3D/ReadVariableOp� conv3d_15/BiasAdd/ReadVariableOp�conv3d_15/Conv3D/ReadVariableOp� conv3d_16/BiasAdd/ReadVariableOp�conv3d_16/Conv3D/ReadVariableOp� conv3d_17/BiasAdd/ReadVariableOp�conv3d_17/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�conv3d_6/BiasAdd/ReadVariableOp�conv3d_6/Conv3D/ReadVariableOp�conv3d_7/BiasAdd/ReadVariableOp�conv3d_7/Conv3D/ReadVariableOp�conv3d_8/BiasAdd/ReadVariableOp�conv3d_8/Conv3D/ReadVariableOp�conv3d_9/BiasAdd/ReadVariableOp�conv3d_9/Conv3D/ReadVariableOp�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_2/Conv3DConv3Dconv3d_1/Relu:activations:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_4/Conv3DConv3Dconv3d_3/Relu:activations:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_6/Conv3DConv3Dconv3d_5/Relu:activations:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_7/Conv3DConv3Dconv3d_6/Relu:activations:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_8/Conv3DConv3Dconv3d_7/Relu:activations:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_8/ReluReluconv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   v
1activity_regularization/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
/activity_regularization/ActivityRegularizer/AbsAbsconv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
3activity_regularization/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*)
value B"                �
/activity_regularization/ActivityRegularizer/SumSum3activity_regularization/ActivityRegularizer/Abs:y:0<activity_regularization/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: v
1activity_regularization/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/activity_regularization/ActivityRegularizer/mulMul:activity_regularization/ActivityRegularizer/mul/x:output:08activity_regularization/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: �
/activity_regularization/ActivityRegularizer/addAddV2:activity_regularization/ActivityRegularizer/Const:output:03activity_regularization/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: |
1activity_regularization/ActivityRegularizer/ShapeShapeconv3d_8/Relu:activations:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDiv3activity_regularization/ActivityRegularizer/add:z:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_9/Conv3DConv3Dconv3d_8/Relu:activations:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_9/ReluReluconv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_10/Conv3DConv3Dconv3d_9/Relu:activations:0'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_10/ReluReluconv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_11/Conv3D/ReadVariableOpReadVariableOp(conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_11/Conv3DConv3Dconv3d_10/Relu:activations:0'conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_11/BiasAdd/ReadVariableOpReadVariableOp)conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_11/BiasAddBiasAddconv3d_11/Conv3D:output:0(conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_11/ReluReluconv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_12/Conv3D/ReadVariableOpReadVariableOp(conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_12/Conv3DConv3Dconv3d_11/Relu:activations:0'conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_12/BiasAdd/ReadVariableOpReadVariableOp)conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_12/BiasAddBiasAddconv3d_12/Conv3D:output:0(conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_12/ReluReluconv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_13/Conv3D/ReadVariableOpReadVariableOp(conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_13/Conv3DConv3Dconv3d_12/Relu:activations:0'conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_13/BiasAdd/ReadVariableOpReadVariableOp)conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_13/BiasAddBiasAddconv3d_13/Conv3D:output:0(conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_13/ReluReluconv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_14/Conv3D/ReadVariableOpReadVariableOp(conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_14/Conv3DConv3Dconv3d_13/Relu:activations:0'conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_14/BiasAdd/ReadVariableOpReadVariableOp)conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_14/BiasAddBiasAddconv3d_14/Conv3D:output:0(conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_14/ReluReluconv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_15/Conv3D/ReadVariableOpReadVariableOp(conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_15/Conv3DConv3Dconv3d_14/Relu:activations:0'conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
 conv3d_15/BiasAdd/ReadVariableOpReadVariableOp)conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
conv3d_15/BiasAddBiasAddconv3d_15/Conv3D:output:0(conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	p
conv3d_15/ReluReluconv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
conv3d_16/Conv3D/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_16/Conv3DConv3Dconv3d_15/Relu:activations:0'conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_16/BiasAdd/ReadVariableOpReadVariableOp)conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_16/BiasAddBiasAddconv3d_16/Conv3D:output:0(conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_16/ReluReluconv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_17/Conv3D/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_17/Conv3DConv3Dconv3d_16/Relu:activations:0'conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_17/BiasAdd/ReadVariableOpReadVariableOp)conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_17/BiasAddBiasAddconv3d_17/Conv3D:output:0(conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   u
IdentityIdentityconv3d_17/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp!^conv3d_11/BiasAdd/ReadVariableOp ^conv3d_11/Conv3D/ReadVariableOp!^conv3d_12/BiasAdd/ReadVariableOp ^conv3d_12/Conv3D/ReadVariableOp!^conv3d_13/BiasAdd/ReadVariableOp ^conv3d_13/Conv3D/ReadVariableOp!^conv3d_14/BiasAdd/ReadVariableOp ^conv3d_14/Conv3D/ReadVariableOp!^conv3d_15/BiasAdd/ReadVariableOp ^conv3d_15/Conv3D/ReadVariableOp!^conv3d_16/BiasAdd/ReadVariableOp ^conv3d_16/Conv3D/ReadVariableOp!^conv3d_17/BiasAdd/ReadVariableOp ^conv3d_17/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2D
 conv3d_10/BiasAdd/ReadVariableOp conv3d_10/BiasAdd/ReadVariableOp2B
conv3d_10/Conv3D/ReadVariableOpconv3d_10/Conv3D/ReadVariableOp2D
 conv3d_11/BiasAdd/ReadVariableOp conv3d_11/BiasAdd/ReadVariableOp2B
conv3d_11/Conv3D/ReadVariableOpconv3d_11/Conv3D/ReadVariableOp2D
 conv3d_12/BiasAdd/ReadVariableOp conv3d_12/BiasAdd/ReadVariableOp2B
conv3d_12/Conv3D/ReadVariableOpconv3d_12/Conv3D/ReadVariableOp2D
 conv3d_13/BiasAdd/ReadVariableOp conv3d_13/BiasAdd/ReadVariableOp2B
conv3d_13/Conv3D/ReadVariableOpconv3d_13/Conv3D/ReadVariableOp2D
 conv3d_14/BiasAdd/ReadVariableOp conv3d_14/BiasAdd/ReadVariableOp2B
conv3d_14/Conv3D/ReadVariableOpconv3d_14/Conv3D/ReadVariableOp2D
 conv3d_15/BiasAdd/ReadVariableOp conv3d_15/BiasAdd/ReadVariableOp2B
conv3d_15/Conv3D/ReadVariableOpconv3d_15/Conv3D/ReadVariableOp2D
 conv3d_16/BiasAdd/ReadVariableOp conv3d_16/BiasAdd/ReadVariableOp2B
conv3d_16/Conv3D/ReadVariableOpconv3d_16/Conv3D/ReadVariableOp2D
 conv3d_17/BiasAdd/ReadVariableOp conv3d_17/BiasAdd/ReadVariableOp2B
conv3d_17/Conv3D/ReadVariableOpconv3d_17/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2B
conv3d_8/BiasAdd/ReadVariableOpconv3d_8/BiasAdd/ReadVariableOp2@
conv3d_8/Conv3D/ReadVariableOpconv3d_8/Conv3D/ReadVariableOp2B
conv3d_9/BiasAdd/ReadVariableOpconv3d_9/BiasAdd/ReadVariableOp2@
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
��
�/
__inference__traced_save_387346
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
(savev2_conv3d_5_bias_read_readvariableop.
*savev2_conv3d_6_kernel_read_readvariableop,
(savev2_conv3d_6_bias_read_readvariableop.
*savev2_conv3d_7_kernel_read_readvariableop,
(savev2_conv3d_7_bias_read_readvariableop.
*savev2_conv3d_8_kernel_read_readvariableop,
(savev2_conv3d_8_bias_read_readvariableop.
*savev2_conv3d_9_kernel_read_readvariableop,
(savev2_conv3d_9_bias_read_readvariableop/
+savev2_conv3d_10_kernel_read_readvariableop-
)savev2_conv3d_10_bias_read_readvariableop/
+savev2_conv3d_11_kernel_read_readvariableop-
)savev2_conv3d_11_bias_read_readvariableop/
+savev2_conv3d_12_kernel_read_readvariableop-
)savev2_conv3d_12_bias_read_readvariableop/
+savev2_conv3d_13_kernel_read_readvariableop-
)savev2_conv3d_13_bias_read_readvariableop/
+savev2_conv3d_14_kernel_read_readvariableop-
)savev2_conv3d_14_bias_read_readvariableop/
+savev2_conv3d_15_kernel_read_readvariableop-
)savev2_conv3d_15_bias_read_readvariableop/
+savev2_conv3d_16_kernel_read_readvariableop-
)savev2_conv3d_16_bias_read_readvariableop/
+savev2_conv3d_17_kernel_read_readvariableop-
)savev2_conv3d_17_bias_read_readvariableop(
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
/savev2_adam_conv3d_5_bias_m_read_readvariableop5
1savev2_adam_conv3d_6_kernel_m_read_readvariableop3
/savev2_adam_conv3d_6_bias_m_read_readvariableop5
1savev2_adam_conv3d_7_kernel_m_read_readvariableop3
/savev2_adam_conv3d_7_bias_m_read_readvariableop5
1savev2_adam_conv3d_8_kernel_m_read_readvariableop3
/savev2_adam_conv3d_8_bias_m_read_readvariableop5
1savev2_adam_conv3d_9_kernel_m_read_readvariableop3
/savev2_adam_conv3d_9_bias_m_read_readvariableop6
2savev2_adam_conv3d_10_kernel_m_read_readvariableop4
0savev2_adam_conv3d_10_bias_m_read_readvariableop6
2savev2_adam_conv3d_11_kernel_m_read_readvariableop4
0savev2_adam_conv3d_11_bias_m_read_readvariableop6
2savev2_adam_conv3d_12_kernel_m_read_readvariableop4
0savev2_adam_conv3d_12_bias_m_read_readvariableop6
2savev2_adam_conv3d_13_kernel_m_read_readvariableop4
0savev2_adam_conv3d_13_bias_m_read_readvariableop6
2savev2_adam_conv3d_14_kernel_m_read_readvariableop4
0savev2_adam_conv3d_14_bias_m_read_readvariableop6
2savev2_adam_conv3d_15_kernel_m_read_readvariableop4
0savev2_adam_conv3d_15_bias_m_read_readvariableop6
2savev2_adam_conv3d_16_kernel_m_read_readvariableop4
0savev2_adam_conv3d_16_bias_m_read_readvariableop6
2savev2_adam_conv3d_17_kernel_m_read_readvariableop4
0savev2_adam_conv3d_17_bias_m_read_readvariableop3
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
/savev2_adam_conv3d_5_bias_v_read_readvariableop5
1savev2_adam_conv3d_6_kernel_v_read_readvariableop3
/savev2_adam_conv3d_6_bias_v_read_readvariableop5
1savev2_adam_conv3d_7_kernel_v_read_readvariableop3
/savev2_adam_conv3d_7_bias_v_read_readvariableop5
1savev2_adam_conv3d_8_kernel_v_read_readvariableop3
/savev2_adam_conv3d_8_bias_v_read_readvariableop5
1savev2_adam_conv3d_9_kernel_v_read_readvariableop3
/savev2_adam_conv3d_9_bias_v_read_readvariableop6
2savev2_adam_conv3d_10_kernel_v_read_readvariableop4
0savev2_adam_conv3d_10_bias_v_read_readvariableop6
2savev2_adam_conv3d_11_kernel_v_read_readvariableop4
0savev2_adam_conv3d_11_bias_v_read_readvariableop6
2savev2_adam_conv3d_12_kernel_v_read_readvariableop4
0savev2_adam_conv3d_12_bias_v_read_readvariableop6
2savev2_adam_conv3d_13_kernel_v_read_readvariableop4
0savev2_adam_conv3d_13_bias_v_read_readvariableop6
2savev2_adam_conv3d_14_kernel_v_read_readvariableop4
0savev2_adam_conv3d_14_bias_v_read_readvariableop6
2savev2_adam_conv3d_15_kernel_v_read_readvariableop4
0savev2_adam_conv3d_15_bias_v_read_readvariableop6
2savev2_adam_conv3d_16_kernel_v_read_readvariableop4
0savev2_adam_conv3d_16_bias_v_read_readvariableop6
2savev2_adam_conv3d_17_kernel_v_read_readvariableop4
0savev2_adam_conv3d_17_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �B
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*�A
value�AB�AtB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*�
value�B�tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop*savev2_conv3d_6_kernel_read_readvariableop(savev2_conv3d_6_bias_read_readvariableop*savev2_conv3d_7_kernel_read_readvariableop(savev2_conv3d_7_bias_read_readvariableop*savev2_conv3d_8_kernel_read_readvariableop(savev2_conv3d_8_bias_read_readvariableop*savev2_conv3d_9_kernel_read_readvariableop(savev2_conv3d_9_bias_read_readvariableop+savev2_conv3d_10_kernel_read_readvariableop)savev2_conv3d_10_bias_read_readvariableop+savev2_conv3d_11_kernel_read_readvariableop)savev2_conv3d_11_bias_read_readvariableop+savev2_conv3d_12_kernel_read_readvariableop)savev2_conv3d_12_bias_read_readvariableop+savev2_conv3d_13_kernel_read_readvariableop)savev2_conv3d_13_bias_read_readvariableop+savev2_conv3d_14_kernel_read_readvariableop)savev2_conv3d_14_bias_read_readvariableop+savev2_conv3d_15_kernel_read_readvariableop)savev2_conv3d_15_bias_read_readvariableop+savev2_conv3d_16_kernel_read_readvariableop)savev2_conv3d_16_bias_read_readvariableop+savev2_conv3d_17_kernel_read_readvariableop)savev2_conv3d_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv3d_kernel_m_read_readvariableop-savev2_adam_conv3d_bias_m_read_readvariableop1savev2_adam_conv3d_1_kernel_m_read_readvariableop/savev2_adam_conv3d_1_bias_m_read_readvariableop1savev2_adam_conv3d_2_kernel_m_read_readvariableop/savev2_adam_conv3d_2_bias_m_read_readvariableop1savev2_adam_conv3d_3_kernel_m_read_readvariableop/savev2_adam_conv3d_3_bias_m_read_readvariableop1savev2_adam_conv3d_4_kernel_m_read_readvariableop/savev2_adam_conv3d_4_bias_m_read_readvariableop1savev2_adam_conv3d_5_kernel_m_read_readvariableop/savev2_adam_conv3d_5_bias_m_read_readvariableop1savev2_adam_conv3d_6_kernel_m_read_readvariableop/savev2_adam_conv3d_6_bias_m_read_readvariableop1savev2_adam_conv3d_7_kernel_m_read_readvariableop/savev2_adam_conv3d_7_bias_m_read_readvariableop1savev2_adam_conv3d_8_kernel_m_read_readvariableop/savev2_adam_conv3d_8_bias_m_read_readvariableop1savev2_adam_conv3d_9_kernel_m_read_readvariableop/savev2_adam_conv3d_9_bias_m_read_readvariableop2savev2_adam_conv3d_10_kernel_m_read_readvariableop0savev2_adam_conv3d_10_bias_m_read_readvariableop2savev2_adam_conv3d_11_kernel_m_read_readvariableop0savev2_adam_conv3d_11_bias_m_read_readvariableop2savev2_adam_conv3d_12_kernel_m_read_readvariableop0savev2_adam_conv3d_12_bias_m_read_readvariableop2savev2_adam_conv3d_13_kernel_m_read_readvariableop0savev2_adam_conv3d_13_bias_m_read_readvariableop2savev2_adam_conv3d_14_kernel_m_read_readvariableop0savev2_adam_conv3d_14_bias_m_read_readvariableop2savev2_adam_conv3d_15_kernel_m_read_readvariableop0savev2_adam_conv3d_15_bias_m_read_readvariableop2savev2_adam_conv3d_16_kernel_m_read_readvariableop0savev2_adam_conv3d_16_bias_m_read_readvariableop2savev2_adam_conv3d_17_kernel_m_read_readvariableop0savev2_adam_conv3d_17_bias_m_read_readvariableop/savev2_adam_conv3d_kernel_v_read_readvariableop-savev2_adam_conv3d_bias_v_read_readvariableop1savev2_adam_conv3d_1_kernel_v_read_readvariableop/savev2_adam_conv3d_1_bias_v_read_readvariableop1savev2_adam_conv3d_2_kernel_v_read_readvariableop/savev2_adam_conv3d_2_bias_v_read_readvariableop1savev2_adam_conv3d_3_kernel_v_read_readvariableop/savev2_adam_conv3d_3_bias_v_read_readvariableop1savev2_adam_conv3d_4_kernel_v_read_readvariableop/savev2_adam_conv3d_4_bias_v_read_readvariableop1savev2_adam_conv3d_5_kernel_v_read_readvariableop/savev2_adam_conv3d_5_bias_v_read_readvariableop1savev2_adam_conv3d_6_kernel_v_read_readvariableop/savev2_adam_conv3d_6_bias_v_read_readvariableop1savev2_adam_conv3d_7_kernel_v_read_readvariableop/savev2_adam_conv3d_7_bias_v_read_readvariableop1savev2_adam_conv3d_8_kernel_v_read_readvariableop/savev2_adam_conv3d_8_bias_v_read_readvariableop1savev2_adam_conv3d_9_kernel_v_read_readvariableop/savev2_adam_conv3d_9_bias_v_read_readvariableop2savev2_adam_conv3d_10_kernel_v_read_readvariableop0savev2_adam_conv3d_10_bias_v_read_readvariableop2savev2_adam_conv3d_11_kernel_v_read_readvariableop0savev2_adam_conv3d_11_bias_v_read_readvariableop2savev2_adam_conv3d_12_kernel_v_read_readvariableop0savev2_adam_conv3d_12_bias_v_read_readvariableop2savev2_adam_conv3d_13_kernel_v_read_readvariableop0savev2_adam_conv3d_13_bias_v_read_readvariableop2savev2_adam_conv3d_14_kernel_v_read_readvariableop0savev2_adam_conv3d_14_bias_v_read_readvariableop2savev2_adam_conv3d_15_kernel_v_read_readvariableop0savev2_adam_conv3d_15_bias_v_read_readvariableop2savev2_adam_conv3d_16_kernel_v_read_readvariableop0savev2_adam_conv3d_16_bias_v_read_readvariableop2savev2_adam_conv3d_17_kernel_v_read_readvariableop0savev2_adam_conv3d_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypesx
v2t	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :::	:	:	::::::::::::::::::::::::::	:	:	:::: : : : : : : :::	:	:	::::::::::::::::::::::::::	:	:	::::::	:	:	::::::::::::::::::::::::::	:	:	:::: 2(
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
:: 

_output_shapes
::0	,
*
_output_shapes
:: 


_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:	:  

_output_shapes
:	:0!,
*
_output_shapes
:	: "

_output_shapes
::0#,
*
_output_shapes
:: $

_output_shapes
::%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :0,,
*
_output_shapes
:: -

_output_shapes
::0.,
*
_output_shapes
:	: /

_output_shapes
:	:00,
*
_output_shapes
:	: 1

_output_shapes
::02,
*
_output_shapes
:: 3

_output_shapes
::04,
*
_output_shapes
:: 5

_output_shapes
::06,
*
_output_shapes
:: 7

_output_shapes
::08,
*
_output_shapes
:: 9

_output_shapes
::0:,
*
_output_shapes
:: ;

_output_shapes
::0<,
*
_output_shapes
:: =

_output_shapes
::0>,
*
_output_shapes
:: ?

_output_shapes
::0@,
*
_output_shapes
:: A

_output_shapes
::0B,
*
_output_shapes
:: C

_output_shapes
::0D,
*
_output_shapes
:: E

_output_shapes
::0F,
*
_output_shapes
:: G

_output_shapes
::0H,
*
_output_shapes
:: I

_output_shapes
::0J,
*
_output_shapes
:	: K

_output_shapes
:	:0L,
*
_output_shapes
:	: M

_output_shapes
::0N,
*
_output_shapes
:: O

_output_shapes
::0P,
*
_output_shapes
:: Q

_output_shapes
::0R,
*
_output_shapes
:	: S

_output_shapes
:	:0T,
*
_output_shapes
:	: U

_output_shapes
::0V,
*
_output_shapes
:: W

_output_shapes
::0X,
*
_output_shapes
:: Y

_output_shapes
::0Z,
*
_output_shapes
:: [

_output_shapes
::0\,
*
_output_shapes
:: ]

_output_shapes
::0^,
*
_output_shapes
:: _

_output_shapes
::0`,
*
_output_shapes
:: a

_output_shapes
::0b,
*
_output_shapes
:: c

_output_shapes
::0d,
*
_output_shapes
:: e

_output_shapes
::0f,
*
_output_shapes
:: g

_output_shapes
::0h,
*
_output_shapes
:: i

_output_shapes
::0j,
*
_output_shapes
:: k

_output_shapes
::0l,
*
_output_shapes
:: m

_output_shapes
::0n,
*
_output_shapes
:	: o

_output_shapes
:	:0p,
*
_output_shapes
:	: q

_output_shapes
::0r,
*
_output_shapes
:: s

_output_shapes
::t

_output_shapes
: 
�
�
)__inference_conv3d_6_layer_call_fn_386716

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_385018{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_8_layer_call_and_return_conditional_losses_386767

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�

&__inference_model_layer_call_fn_386299

inputs%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:	

unknown_30:	(

unknown_31:	

unknown_32:(

unknown_33:

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:���������   : *F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_385696{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_5_layer_call_fn_386696

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_385001{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
E__inference_conv3d_14_layer_call_and_return_conditional_losses_386911

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
*__inference_conv3d_10_layer_call_fn_386820

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_385100{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_5_layer_call_and_return_conditional_losses_385001

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
V
?__inference_activity_regularization_activity_regularizer_384898
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
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:I
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
�
�
D__inference_conv3d_5_layer_call_and_return_conditional_losses_386707

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
*__inference_conv3d_17_layer_call_fn_386960

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_385218{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
*__inference_conv3d_15_layer_call_fn_386920

inputs%
unknown:	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_385185{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385062

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_8_layer_call_and_return_conditional_losses_385052

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�q
�
A__inference_model_layer_call_and_return_conditional_losses_385954
input_1+
conv3d_385853:
conv3d_385855:-
conv3d_1_385858:	
conv3d_1_385860:	-
conv3d_2_385863:	
conv3d_2_385865:-
conv3d_3_385868:
conv3d_3_385870:-
conv3d_4_385873:
conv3d_4_385875:-
conv3d_5_385878:
conv3d_5_385880:-
conv3d_6_385883:
conv3d_6_385885:-
conv3d_7_385888:
conv3d_7_385890:-
conv3d_8_385893:
conv3d_8_385895:-
conv3d_9_385907:
conv3d_9_385909:.
conv3d_10_385912:
conv3d_10_385914:.
conv3d_11_385917:
conv3d_11_385919:.
conv3d_12_385922:
conv3d_12_385924:.
conv3d_13_385927:
conv3d_13_385929:.
conv3d_14_385932:
conv3d_14_385934:.
conv3d_15_385937:	
conv3d_15_385939:	.
conv3d_16_385942:	
conv3d_16_385944:.
conv3d_17_385947:
conv3d_17_385949:
identity

identity_1��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall�!conv3d_10/StatefulPartitionedCall�!conv3d_11/StatefulPartitionedCall�!conv3d_12/StatefulPartitionedCall�!conv3d_13/StatefulPartitionedCall�!conv3d_14/StatefulPartitionedCall�!conv3d_15/StatefulPartitionedCall�!conv3d_16/StatefulPartitionedCall�!conv3d_17/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall� conv3d_8/StatefulPartitionedCall� conv3d_9/StatefulPartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_385853conv3d_385855*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_384916�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_385858conv3d_1_385860*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_384933�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_385863conv3d_2_385865*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_384950�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_385868conv3d_3_385870*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_384967�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_385873conv3d_4_385875*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_384984�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_385878conv3d_5_385880*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_385001�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_385883conv3d_6_385885*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_385018�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_385888conv3d_7_385890*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_385035�
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_385893conv3d_8_385895*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_385052�
'activity_regularization/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385062�
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
GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_384898�
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_9_385907conv3d_9_385909*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_385083�
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0conv3d_10_385912conv3d_10_385914*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_385100�
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_385917conv3d_11_385919*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_385117�
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0conv3d_12_385922conv3d_12_385924*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_385134�
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_385927conv3d_13_385929*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_385151�
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0conv3d_14_385932conv3d_14_385934*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_385168�
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0conv3d_15_385937conv3d_15_385939*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_385185�
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0conv3d_16_385942conv3d_16_385944*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_385202�
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_385947conv3d_17_385949*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_385218�
IdentityIdentity*conv3d_17/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
E__inference_conv3d_15_layer_call_and_return_conditional_losses_386931

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
E__inference_conv3d_10_layer_call_and_return_conditional_losses_385100

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
B__inference_conv3d_layer_call_and_return_conditional_losses_386607

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_3_layer_call_and_return_conditional_losses_386667

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
E__inference_conv3d_10_layer_call_and_return_conditional_losses_386831

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
'__inference_conv3d_layer_call_fn_386596

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_384916{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_3_layer_call_and_return_conditional_losses_384967

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
T
8__inference_activity_regularization_layer_call_fn_386777

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385406l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�

�
E__inference_conv3d_17_layer_call_and_return_conditional_losses_385218

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_2_layer_call_fn_386636

inputs%
unknown:	
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_384950{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�
�
)__inference_conv3d_9_layer_call_fn_386800

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_385083{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_7_layer_call_and_return_conditional_losses_386747

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_9_layer_call_and_return_conditional_losses_386811

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�q
�
A__inference_model_layer_call_and_return_conditional_losses_385696

inputs+
conv3d_385595:
conv3d_385597:-
conv3d_1_385600:	
conv3d_1_385602:	-
conv3d_2_385605:	
conv3d_2_385607:-
conv3d_3_385610:
conv3d_3_385612:-
conv3d_4_385615:
conv3d_4_385617:-
conv3d_5_385620:
conv3d_5_385622:-
conv3d_6_385625:
conv3d_6_385627:-
conv3d_7_385630:
conv3d_7_385632:-
conv3d_8_385635:
conv3d_8_385637:-
conv3d_9_385649:
conv3d_9_385651:.
conv3d_10_385654:
conv3d_10_385656:.
conv3d_11_385659:
conv3d_11_385661:.
conv3d_12_385664:
conv3d_12_385666:.
conv3d_13_385669:
conv3d_13_385671:.
conv3d_14_385674:
conv3d_14_385676:.
conv3d_15_385679:	
conv3d_15_385681:	.
conv3d_16_385684:	
conv3d_16_385686:.
conv3d_17_385689:
conv3d_17_385691:
identity

identity_1��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall�!conv3d_10/StatefulPartitionedCall�!conv3d_11/StatefulPartitionedCall�!conv3d_12/StatefulPartitionedCall�!conv3d_13/StatefulPartitionedCall�!conv3d_14/StatefulPartitionedCall�!conv3d_15/StatefulPartitionedCall�!conv3d_16/StatefulPartitionedCall�!conv3d_17/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall� conv3d_8/StatefulPartitionedCall� conv3d_9/StatefulPartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_385595conv3d_385597*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_384916�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_385600conv3d_1_385602*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_384933�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_385605conv3d_2_385607*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_384950�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_385610conv3d_3_385612*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_384967�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_385615conv3d_4_385617*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_384984�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_385620conv3d_5_385622*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_385001�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_385625conv3d_6_385627*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_385018�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_385630conv3d_7_385632*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_385035�
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_385635conv3d_8_385637*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_385052�
'activity_regularization/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385406�
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
GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_384898�
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_9_385649conv3d_9_385651*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_385083�
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0conv3d_10_385654conv3d_10_385656*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_385100�
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_385659conv3d_11_385661*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_385117�
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0conv3d_12_385664conv3d_12_385666*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_385134�
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_385669conv3d_13_385671*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_385151�
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0conv3d_14_385674conv3d_14_385676*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_385168�
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0conv3d_15_385679conv3d_15_385681*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_385185�
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0conv3d_16_385684conv3d_16_385686*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_385202�
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_385689conv3d_17_385691*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_385218�
IdentityIdentity*conv3d_17/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_4_layer_call_and_return_conditional_losses_384984

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
E__inference_conv3d_13_layer_call_and_return_conditional_losses_386891

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_1_layer_call_and_return_conditional_losses_384933

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_9_layer_call_and_return_conditional_losses_385083

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_384883
input_1I
+model_conv3d_conv3d_readvariableop_resource::
,model_conv3d_biasadd_readvariableop_resource:K
-model_conv3d_1_conv3d_readvariableop_resource:	<
.model_conv3d_1_biasadd_readvariableop_resource:	K
-model_conv3d_2_conv3d_readvariableop_resource:	<
.model_conv3d_2_biasadd_readvariableop_resource:K
-model_conv3d_3_conv3d_readvariableop_resource:<
.model_conv3d_3_biasadd_readvariableop_resource:K
-model_conv3d_4_conv3d_readvariableop_resource:<
.model_conv3d_4_biasadd_readvariableop_resource:K
-model_conv3d_5_conv3d_readvariableop_resource:<
.model_conv3d_5_biasadd_readvariableop_resource:K
-model_conv3d_6_conv3d_readvariableop_resource:<
.model_conv3d_6_biasadd_readvariableop_resource:K
-model_conv3d_7_conv3d_readvariableop_resource:<
.model_conv3d_7_biasadd_readvariableop_resource:K
-model_conv3d_8_conv3d_readvariableop_resource:<
.model_conv3d_8_biasadd_readvariableop_resource:K
-model_conv3d_9_conv3d_readvariableop_resource:<
.model_conv3d_9_biasadd_readvariableop_resource:L
.model_conv3d_10_conv3d_readvariableop_resource:=
/model_conv3d_10_biasadd_readvariableop_resource:L
.model_conv3d_11_conv3d_readvariableop_resource:=
/model_conv3d_11_biasadd_readvariableop_resource:L
.model_conv3d_12_conv3d_readvariableop_resource:=
/model_conv3d_12_biasadd_readvariableop_resource:L
.model_conv3d_13_conv3d_readvariableop_resource:=
/model_conv3d_13_biasadd_readvariableop_resource:L
.model_conv3d_14_conv3d_readvariableop_resource:=
/model_conv3d_14_biasadd_readvariableop_resource:L
.model_conv3d_15_conv3d_readvariableop_resource:	=
/model_conv3d_15_biasadd_readvariableop_resource:	L
.model_conv3d_16_conv3d_readvariableop_resource:	=
/model_conv3d_16_biasadd_readvariableop_resource:L
.model_conv3d_17_conv3d_readvariableop_resource:=
/model_conv3d_17_biasadd_readvariableop_resource:
identity��#model/conv3d/BiasAdd/ReadVariableOp�"model/conv3d/Conv3D/ReadVariableOp�%model/conv3d_1/BiasAdd/ReadVariableOp�$model/conv3d_1/Conv3D/ReadVariableOp�&model/conv3d_10/BiasAdd/ReadVariableOp�%model/conv3d_10/Conv3D/ReadVariableOp�&model/conv3d_11/BiasAdd/ReadVariableOp�%model/conv3d_11/Conv3D/ReadVariableOp�&model/conv3d_12/BiasAdd/ReadVariableOp�%model/conv3d_12/Conv3D/ReadVariableOp�&model/conv3d_13/BiasAdd/ReadVariableOp�%model/conv3d_13/Conv3D/ReadVariableOp�&model/conv3d_14/BiasAdd/ReadVariableOp�%model/conv3d_14/Conv3D/ReadVariableOp�&model/conv3d_15/BiasAdd/ReadVariableOp�%model/conv3d_15/Conv3D/ReadVariableOp�&model/conv3d_16/BiasAdd/ReadVariableOp�%model/conv3d_16/Conv3D/ReadVariableOp�&model/conv3d_17/BiasAdd/ReadVariableOp�%model/conv3d_17/Conv3D/ReadVariableOp�%model/conv3d_2/BiasAdd/ReadVariableOp�$model/conv3d_2/Conv3D/ReadVariableOp�%model/conv3d_3/BiasAdd/ReadVariableOp�$model/conv3d_3/Conv3D/ReadVariableOp�%model/conv3d_4/BiasAdd/ReadVariableOp�$model/conv3d_4/Conv3D/ReadVariableOp�%model/conv3d_5/BiasAdd/ReadVariableOp�$model/conv3d_5/Conv3D/ReadVariableOp�%model/conv3d_6/BiasAdd/ReadVariableOp�$model/conv3d_6/Conv3D/ReadVariableOp�%model/conv3d_7/BiasAdd/ReadVariableOp�$model/conv3d_7/Conv3D/ReadVariableOp�%model/conv3d_8/BiasAdd/ReadVariableOp�$model/conv3d_8/Conv3D/ReadVariableOp�%model/conv3d_9/BiasAdd/ReadVariableOp�$model/conv3d_9/Conv3D/ReadVariableOp�
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d/Conv3DConv3Dinput_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   v
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
model/conv3d_1/Conv3DConv3Dmodel/conv3d/Relu:activations:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	z
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
model/conv3d_2/Conv3DConv3D!model/conv3d_1/Relu:activations:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_3/Conv3DConv3D!model/conv3d_2/Relu:activations:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_3/ReluRelumodel/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_4/Conv3DConv3D!model/conv3d_3/Relu:activations:0,model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_4/BiasAddBiasAddmodel/conv3d_4/Conv3D:output:0-model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_4/ReluRelumodel/conv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_5/Conv3DConv3D!model/conv3d_4/Relu:activations:0,model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_5/BiasAddBiasAddmodel/conv3d_5/Conv3D:output:0-model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_5/ReluRelumodel/conv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_6/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_6/Conv3DConv3D!model/conv3d_5/Relu:activations:0,model/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_6/BiasAddBiasAddmodel/conv3d_6/Conv3D:output:0-model/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_6/ReluRelumodel/conv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_7/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_7/Conv3DConv3D!model/conv3d_6/Relu:activations:0,model/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_7/BiasAddBiasAddmodel/conv3d_7/Conv3D:output:0-model/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_7/ReluRelumodel/conv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
$model/conv3d_8/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_8/Conv3DConv3D!model/conv3d_7/Relu:activations:0,model/conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_8/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_8/BiasAddBiasAddmodel/conv3d_8/Conv3D:output:0-model/conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_8/ReluRelumodel/conv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   |
7model/activity_regularization/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
5model/activity_regularization/ActivityRegularizer/AbsAbs!model/conv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
9model/activity_regularization/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*)
value B"                �
5model/activity_regularization/ActivityRegularizer/SumSum9model/activity_regularization/ActivityRegularizer/Abs:y:0Bmodel/activity_regularization/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: |
7model/activity_regularization/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
5model/activity_regularization/ActivityRegularizer/mulMul@model/activity_regularization/ActivityRegularizer/mul/x:output:0>model/activity_regularization/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: �
5model/activity_regularization/ActivityRegularizer/addAddV2@model/activity_regularization/ActivityRegularizer/Const:output:09model/activity_regularization/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: �
7model/activity_regularization/ActivityRegularizer/ShapeShape!model/conv3d_8/Relu:activations:0*
T0*
_output_shapes
:�
Emodel/activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Gmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?model/activity_regularization/ActivityRegularizer/strided_sliceStridedSlice@model/activity_regularization/ActivityRegularizer/Shape:output:0Nmodel/activity_regularization/ActivityRegularizer/strided_slice/stack:output:0Pmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Pmodel/activity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
6model/activity_regularization/ActivityRegularizer/CastCastHmodel/activity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
9model/activity_regularization/ActivityRegularizer/truedivRealDiv9model/activity_regularization/ActivityRegularizer/add:z:0:model/activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
$model/conv3d_9/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_9/Conv3DConv3D!model/conv3d_8/Relu:activations:0,model/conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
%model/conv3d_9/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_9/BiasAddBiasAddmodel/conv3d_9/Conv3D:output:0-model/conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   z
model/conv3d_9/ReluRelumodel/conv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_10/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_10/Conv3DConv3D!model/conv3d_9/Relu:activations:0-model/conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_10/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_10/BiasAddBiasAddmodel/conv3d_10/Conv3D:output:0.model/conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_10/ReluRelu model/conv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_11/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_11/Conv3DConv3D"model/conv3d_10/Relu:activations:0-model/conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_11/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_11/BiasAddBiasAddmodel/conv3d_11/Conv3D:output:0.model/conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_11/ReluRelu model/conv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_12/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_12/Conv3DConv3D"model/conv3d_11/Relu:activations:0-model/conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_12/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_12/BiasAddBiasAddmodel/conv3d_12/Conv3D:output:0.model/conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_12/ReluRelu model/conv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_13/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_13/Conv3DConv3D"model/conv3d_12/Relu:activations:0-model/conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_13/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_13/BiasAddBiasAddmodel/conv3d_13/Conv3D:output:0.model/conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_13/ReluRelu model/conv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_14/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_14/Conv3DConv3D"model/conv3d_13/Relu:activations:0-model/conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_14/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_14/BiasAddBiasAddmodel/conv3d_14/Conv3D:output:0.model/conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_14/ReluRelu model/conv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_15/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
model/conv3d_15/Conv3DConv3D"model/conv3d_14/Relu:activations:0-model/conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
&model/conv3d_15/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
model/conv3d_15/BiasAddBiasAddmodel/conv3d_15/Conv3D:output:0.model/conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	|
model/conv3d_15/ReluRelu model/conv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
%model/conv3d_16/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
model/conv3d_16/Conv3DConv3D"model/conv3d_15/Relu:activations:0-model/conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_16/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_16/BiasAddBiasAddmodel/conv3d_16/Conv3D:output:0.model/conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   |
model/conv3d_16/ReluRelu model/conv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
%model/conv3d_17/Conv3D/ReadVariableOpReadVariableOp.model_conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_17/Conv3DConv3D"model/conv3d_16/Relu:activations:0-model/conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
&model/conv3d_17/BiasAdd/ReadVariableOpReadVariableOp/model_conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_17/BiasAddBiasAddmodel/conv3d_17/Conv3D:output:0.model/conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   {
IdentityIdentity model/conv3d_17/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   �
NoOpNoOp$^model/conv3d/BiasAdd/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp'^model/conv3d_10/BiasAdd/ReadVariableOp&^model/conv3d_10/Conv3D/ReadVariableOp'^model/conv3d_11/BiasAdd/ReadVariableOp&^model/conv3d_11/Conv3D/ReadVariableOp'^model/conv3d_12/BiasAdd/ReadVariableOp&^model/conv3d_12/Conv3D/ReadVariableOp'^model/conv3d_13/BiasAdd/ReadVariableOp&^model/conv3d_13/Conv3D/ReadVariableOp'^model/conv3d_14/BiasAdd/ReadVariableOp&^model/conv3d_14/Conv3D/ReadVariableOp'^model/conv3d_15/BiasAdd/ReadVariableOp&^model/conv3d_15/Conv3D/ReadVariableOp'^model/conv3d_16/BiasAdd/ReadVariableOp&^model/conv3d_16/Conv3D/ReadVariableOp'^model/conv3d_17/BiasAdd/ReadVariableOp&^model/conv3d_17/Conv3D/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp&^model/conv3d_3/BiasAdd/ReadVariableOp%^model/conv3d_3/Conv3D/ReadVariableOp&^model/conv3d_4/BiasAdd/ReadVariableOp%^model/conv3d_4/Conv3D/ReadVariableOp&^model/conv3d_5/BiasAdd/ReadVariableOp%^model/conv3d_5/Conv3D/ReadVariableOp&^model/conv3d_6/BiasAdd/ReadVariableOp%^model/conv3d_6/Conv3D/ReadVariableOp&^model/conv3d_7/BiasAdd/ReadVariableOp%^model/conv3d_7/Conv3D/ReadVariableOp&^model/conv3d_8/BiasAdd/ReadVariableOp%^model/conv3d_8/Conv3D/ReadVariableOp&^model/conv3d_9/BiasAdd/ReadVariableOp%^model/conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2P
&model/conv3d_10/BiasAdd/ReadVariableOp&model/conv3d_10/BiasAdd/ReadVariableOp2N
%model/conv3d_10/Conv3D/ReadVariableOp%model/conv3d_10/Conv3D/ReadVariableOp2P
&model/conv3d_11/BiasAdd/ReadVariableOp&model/conv3d_11/BiasAdd/ReadVariableOp2N
%model/conv3d_11/Conv3D/ReadVariableOp%model/conv3d_11/Conv3D/ReadVariableOp2P
&model/conv3d_12/BiasAdd/ReadVariableOp&model/conv3d_12/BiasAdd/ReadVariableOp2N
%model/conv3d_12/Conv3D/ReadVariableOp%model/conv3d_12/Conv3D/ReadVariableOp2P
&model/conv3d_13/BiasAdd/ReadVariableOp&model/conv3d_13/BiasAdd/ReadVariableOp2N
%model/conv3d_13/Conv3D/ReadVariableOp%model/conv3d_13/Conv3D/ReadVariableOp2P
&model/conv3d_14/BiasAdd/ReadVariableOp&model/conv3d_14/BiasAdd/ReadVariableOp2N
%model/conv3d_14/Conv3D/ReadVariableOp%model/conv3d_14/Conv3D/ReadVariableOp2P
&model/conv3d_15/BiasAdd/ReadVariableOp&model/conv3d_15/BiasAdd/ReadVariableOp2N
%model/conv3d_15/Conv3D/ReadVariableOp%model/conv3d_15/Conv3D/ReadVariableOp2P
&model/conv3d_16/BiasAdd/ReadVariableOp&model/conv3d_16/BiasAdd/ReadVariableOp2N
%model/conv3d_16/Conv3D/ReadVariableOp%model/conv3d_16/Conv3D/ReadVariableOp2P
&model/conv3d_17/BiasAdd/ReadVariableOp&model/conv3d_17/BiasAdd/ReadVariableOp2N
%model/conv3d_17/Conv3D/ReadVariableOp%model/conv3d_17/Conv3D/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2N
%model/conv3d_3/BiasAdd/ReadVariableOp%model/conv3d_3/BiasAdd/ReadVariableOp2L
$model/conv3d_3/Conv3D/ReadVariableOp$model/conv3d_3/Conv3D/ReadVariableOp2N
%model/conv3d_4/BiasAdd/ReadVariableOp%model/conv3d_4/BiasAdd/ReadVariableOp2L
$model/conv3d_4/Conv3D/ReadVariableOp$model/conv3d_4/Conv3D/ReadVariableOp2N
%model/conv3d_5/BiasAdd/ReadVariableOp%model/conv3d_5/BiasAdd/ReadVariableOp2L
$model/conv3d_5/Conv3D/ReadVariableOp$model/conv3d_5/Conv3D/ReadVariableOp2N
%model/conv3d_6/BiasAdd/ReadVariableOp%model/conv3d_6/BiasAdd/ReadVariableOp2L
$model/conv3d_6/Conv3D/ReadVariableOp$model/conv3d_6/Conv3D/ReadVariableOp2N
%model/conv3d_7/BiasAdd/ReadVariableOp%model/conv3d_7/BiasAdd/ReadVariableOp2L
$model/conv3d_7/Conv3D/ReadVariableOp$model/conv3d_7/Conv3D/ReadVariableOp2N
%model/conv3d_8/BiasAdd/ReadVariableOp%model/conv3d_8/BiasAdd/ReadVariableOp2L
$model/conv3d_8/Conv3D/ReadVariableOp$model/conv3d_8/Conv3D/ReadVariableOp2N
%model/conv3d_9/BiasAdd/ReadVariableOp%model/conv3d_9/BiasAdd/ReadVariableOp2L
$model/conv3d_9/Conv3D/ReadVariableOp$model/conv3d_9/Conv3D/ReadVariableOp:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385406

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
T
8__inference_activity_regularization_layer_call_fn_386772

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385062l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_2_layer_call_and_return_conditional_losses_384950

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�	
�
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_386784

inputs
identity

identity_1�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385062�
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
GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_384898l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   S

Identity_1IdentityPartitionedCall_1:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
��
�
A__inference_model_layer_call_and_return_conditional_losses_386587

inputsC
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:	6
(conv3d_1_biasadd_readvariableop_resource:	E
'conv3d_2_conv3d_readvariableop_resource:	6
(conv3d_2_biasadd_readvariableop_resource:E
'conv3d_3_conv3d_readvariableop_resource:6
(conv3d_3_biasadd_readvariableop_resource:E
'conv3d_4_conv3d_readvariableop_resource:6
(conv3d_4_biasadd_readvariableop_resource:E
'conv3d_5_conv3d_readvariableop_resource:6
(conv3d_5_biasadd_readvariableop_resource:E
'conv3d_6_conv3d_readvariableop_resource:6
(conv3d_6_biasadd_readvariableop_resource:E
'conv3d_7_conv3d_readvariableop_resource:6
(conv3d_7_biasadd_readvariableop_resource:E
'conv3d_8_conv3d_readvariableop_resource:6
(conv3d_8_biasadd_readvariableop_resource:E
'conv3d_9_conv3d_readvariableop_resource:6
(conv3d_9_biasadd_readvariableop_resource:F
(conv3d_10_conv3d_readvariableop_resource:7
)conv3d_10_biasadd_readvariableop_resource:F
(conv3d_11_conv3d_readvariableop_resource:7
)conv3d_11_biasadd_readvariableop_resource:F
(conv3d_12_conv3d_readvariableop_resource:7
)conv3d_12_biasadd_readvariableop_resource:F
(conv3d_13_conv3d_readvariableop_resource:7
)conv3d_13_biasadd_readvariableop_resource:F
(conv3d_14_conv3d_readvariableop_resource:7
)conv3d_14_biasadd_readvariableop_resource:F
(conv3d_15_conv3d_readvariableop_resource:	7
)conv3d_15_biasadd_readvariableop_resource:	F
(conv3d_16_conv3d_readvariableop_resource:	7
)conv3d_16_biasadd_readvariableop_resource:F
(conv3d_17_conv3d_readvariableop_resource:7
)conv3d_17_biasadd_readvariableop_resource:
identity

identity_1��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp� conv3d_10/BiasAdd/ReadVariableOp�conv3d_10/Conv3D/ReadVariableOp� conv3d_11/BiasAdd/ReadVariableOp�conv3d_11/Conv3D/ReadVariableOp� conv3d_12/BiasAdd/ReadVariableOp�conv3d_12/Conv3D/ReadVariableOp� conv3d_13/BiasAdd/ReadVariableOp�conv3d_13/Conv3D/ReadVariableOp� conv3d_14/BiasAdd/ReadVariableOp�conv3d_14/Conv3D/ReadVariableOp� conv3d_15/BiasAdd/ReadVariableOp�conv3d_15/Conv3D/ReadVariableOp� conv3d_16/BiasAdd/ReadVariableOp�conv3d_16/Conv3D/ReadVariableOp� conv3d_17/BiasAdd/ReadVariableOp�conv3d_17/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�conv3d_6/BiasAdd/ReadVariableOp�conv3d_6/Conv3D/ReadVariableOp�conv3d_7/BiasAdd/ReadVariableOp�conv3d_7/Conv3D/ReadVariableOp�conv3d_8/BiasAdd/ReadVariableOp�conv3d_8/Conv3D/ReadVariableOp�conv3d_9/BiasAdd/ReadVariableOp�conv3d_9/Conv3D/ReadVariableOp�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_2/Conv3DConv3Dconv3d_1/Relu:activations:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_4/Conv3DConv3Dconv3d_3/Relu:activations:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_6/Conv3DConv3Dconv3d_5/Relu:activations:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_7/Conv3DConv3Dconv3d_6/Relu:activations:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_8/Conv3D/ReadVariableOpReadVariableOp'conv3d_8_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_8/Conv3DConv3Dconv3d_7/Relu:activations:0&conv3d_8/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_8/BiasAdd/ReadVariableOpReadVariableOp(conv3d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_8/BiasAddBiasAddconv3d_8/Conv3D:output:0'conv3d_8/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_8/ReluReluconv3d_8/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   v
1activity_regularization/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
/activity_regularization/ActivityRegularizer/AbsAbsconv3d_8/Relu:activations:0*
T0*3
_output_shapes!
:���������   �
3activity_regularization/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*)
value B"                �
/activity_regularization/ActivityRegularizer/SumSum3activity_regularization/ActivityRegularizer/Abs:y:0<activity_regularization/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: v
1activity_regularization/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/activity_regularization/ActivityRegularizer/mulMul:activity_regularization/ActivityRegularizer/mul/x:output:08activity_regularization/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: �
/activity_regularization/ActivityRegularizer/addAddV2:activity_regularization/ActivityRegularizer/Const:output:03activity_regularization/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: |
1activity_regularization/ActivityRegularizer/ShapeShapeconv3d_8/Relu:activations:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDiv3activity_regularization/ActivityRegularizer/add:z:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
conv3d_9/Conv3D/ReadVariableOpReadVariableOp'conv3d_9_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_9/Conv3DConv3Dconv3d_8/Relu:activations:0&conv3d_9/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
conv3d_9/BiasAdd/ReadVariableOpReadVariableOp(conv3d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_9/BiasAddBiasAddconv3d_9/Conv3D:output:0'conv3d_9/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   n
conv3d_9/ReluReluconv3d_9/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_10/Conv3D/ReadVariableOpReadVariableOp(conv3d_10_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_10/Conv3DConv3Dconv3d_9/Relu:activations:0'conv3d_10/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_10/BiasAdd/ReadVariableOpReadVariableOp)conv3d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_10/BiasAddBiasAddconv3d_10/Conv3D:output:0(conv3d_10/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_10/ReluReluconv3d_10/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_11/Conv3D/ReadVariableOpReadVariableOp(conv3d_11_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_11/Conv3DConv3Dconv3d_10/Relu:activations:0'conv3d_11/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_11/BiasAdd/ReadVariableOpReadVariableOp)conv3d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_11/BiasAddBiasAddconv3d_11/Conv3D:output:0(conv3d_11/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_11/ReluReluconv3d_11/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_12/Conv3D/ReadVariableOpReadVariableOp(conv3d_12_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_12/Conv3DConv3Dconv3d_11/Relu:activations:0'conv3d_12/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_12/BiasAdd/ReadVariableOpReadVariableOp)conv3d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_12/BiasAddBiasAddconv3d_12/Conv3D:output:0(conv3d_12/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_12/ReluReluconv3d_12/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_13/Conv3D/ReadVariableOpReadVariableOp(conv3d_13_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_13/Conv3DConv3Dconv3d_12/Relu:activations:0'conv3d_13/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_13/BiasAdd/ReadVariableOpReadVariableOp)conv3d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_13/BiasAddBiasAddconv3d_13/Conv3D:output:0(conv3d_13/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_13/ReluReluconv3d_13/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_14/Conv3D/ReadVariableOpReadVariableOp(conv3d_14_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_14/Conv3DConv3Dconv3d_13/Relu:activations:0'conv3d_14/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_14/BiasAdd/ReadVariableOpReadVariableOp)conv3d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_14/BiasAddBiasAddconv3d_14/Conv3D:output:0(conv3d_14/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_14/ReluReluconv3d_14/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_15/Conv3D/ReadVariableOpReadVariableOp(conv3d_15_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_15/Conv3DConv3Dconv3d_14/Relu:activations:0'conv3d_15/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
�
 conv3d_15/BiasAdd/ReadVariableOpReadVariableOp)conv3d_15_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
conv3d_15/BiasAddBiasAddconv3d_15/Conv3D:output:0(conv3d_15/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	p
conv3d_15/ReluReluconv3d_15/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   	�
conv3d_16/Conv3D/ReadVariableOpReadVariableOp(conv3d_16_conv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
conv3d_16/Conv3DConv3Dconv3d_15/Relu:activations:0'conv3d_16/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_16/BiasAdd/ReadVariableOpReadVariableOp)conv3d_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_16/BiasAddBiasAddconv3d_16/Conv3D:output:0(conv3d_16/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   p
conv3d_16/ReluReluconv3d_16/BiasAdd:output:0*
T0*3
_output_shapes!
:���������   �
conv3d_17/Conv3D/ReadVariableOpReadVariableOp(conv3d_17_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_17/Conv3DConv3Dconv3d_16/Relu:activations:0'conv3d_17/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
�
 conv3d_17/BiasAdd/ReadVariableOpReadVariableOp)conv3d_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_17/BiasAddBiasAddconv3d_17/Conv3D:output:0(conv3d_17/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   u
IdentityIdentityconv3d_17/BiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp!^conv3d_10/BiasAdd/ReadVariableOp ^conv3d_10/Conv3D/ReadVariableOp!^conv3d_11/BiasAdd/ReadVariableOp ^conv3d_11/Conv3D/ReadVariableOp!^conv3d_12/BiasAdd/ReadVariableOp ^conv3d_12/Conv3D/ReadVariableOp!^conv3d_13/BiasAdd/ReadVariableOp ^conv3d_13/Conv3D/ReadVariableOp!^conv3d_14/BiasAdd/ReadVariableOp ^conv3d_14/Conv3D/ReadVariableOp!^conv3d_15/BiasAdd/ReadVariableOp ^conv3d_15/Conv3D/ReadVariableOp!^conv3d_16/BiasAdd/ReadVariableOp ^conv3d_16/Conv3D/ReadVariableOp!^conv3d_17/BiasAdd/ReadVariableOp ^conv3d_17/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp ^conv3d_8/BiasAdd/ReadVariableOp^conv3d_8/Conv3D/ReadVariableOp ^conv3d_9/BiasAdd/ReadVariableOp^conv3d_9/Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2D
 conv3d_10/BiasAdd/ReadVariableOp conv3d_10/BiasAdd/ReadVariableOp2B
conv3d_10/Conv3D/ReadVariableOpconv3d_10/Conv3D/ReadVariableOp2D
 conv3d_11/BiasAdd/ReadVariableOp conv3d_11/BiasAdd/ReadVariableOp2B
conv3d_11/Conv3D/ReadVariableOpconv3d_11/Conv3D/ReadVariableOp2D
 conv3d_12/BiasAdd/ReadVariableOp conv3d_12/BiasAdd/ReadVariableOp2B
conv3d_12/Conv3D/ReadVariableOpconv3d_12/Conv3D/ReadVariableOp2D
 conv3d_13/BiasAdd/ReadVariableOp conv3d_13/BiasAdd/ReadVariableOp2B
conv3d_13/Conv3D/ReadVariableOpconv3d_13/Conv3D/ReadVariableOp2D
 conv3d_14/BiasAdd/ReadVariableOp conv3d_14/BiasAdd/ReadVariableOp2B
conv3d_14/Conv3D/ReadVariableOpconv3d_14/Conv3D/ReadVariableOp2D
 conv3d_15/BiasAdd/ReadVariableOp conv3d_15/BiasAdd/ReadVariableOp2B
conv3d_15/Conv3D/ReadVariableOpconv3d_15/Conv3D/ReadVariableOp2D
 conv3d_16/BiasAdd/ReadVariableOp conv3d_16/BiasAdd/ReadVariableOp2B
conv3d_16/Conv3D/ReadVariableOpconv3d_16/Conv3D/ReadVariableOp2D
 conv3d_17/BiasAdd/ReadVariableOp conv3d_17/BiasAdd/ReadVariableOp2B
conv3d_17/Conv3D/ReadVariableOpconv3d_17/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2B
conv3d_8/BiasAdd/ReadVariableOpconv3d_8/BiasAdd/ReadVariableOp2@
conv3d_8/Conv3D/ReadVariableOpconv3d_8/Conv3D/ReadVariableOp2B
conv3d_9/BiasAdd/ReadVariableOpconv3d_9/BiasAdd/ReadVariableOp2@
conv3d_9/Conv3D/ReadVariableOpconv3d_9/Conv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�q
�
A__inference_model_layer_call_and_return_conditional_losses_386058
input_1+
conv3d_385957:
conv3d_385959:-
conv3d_1_385962:	
conv3d_1_385964:	-
conv3d_2_385967:	
conv3d_2_385969:-
conv3d_3_385972:
conv3d_3_385974:-
conv3d_4_385977:
conv3d_4_385979:-
conv3d_5_385982:
conv3d_5_385984:-
conv3d_6_385987:
conv3d_6_385989:-
conv3d_7_385992:
conv3d_7_385994:-
conv3d_8_385997:
conv3d_8_385999:-
conv3d_9_386011:
conv3d_9_386013:.
conv3d_10_386016:
conv3d_10_386018:.
conv3d_11_386021:
conv3d_11_386023:.
conv3d_12_386026:
conv3d_12_386028:.
conv3d_13_386031:
conv3d_13_386033:.
conv3d_14_386036:
conv3d_14_386038:.
conv3d_15_386041:	
conv3d_15_386043:	.
conv3d_16_386046:	
conv3d_16_386048:.
conv3d_17_386051:
conv3d_17_386053:
identity

identity_1��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall�!conv3d_10/StatefulPartitionedCall�!conv3d_11/StatefulPartitionedCall�!conv3d_12/StatefulPartitionedCall�!conv3d_13/StatefulPartitionedCall�!conv3d_14/StatefulPartitionedCall�!conv3d_15/StatefulPartitionedCall�!conv3d_16/StatefulPartitionedCall�!conv3d_17/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall� conv3d_8/StatefulPartitionedCall� conv3d_9/StatefulPartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_385957conv3d_385959*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_384916�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_385962conv3d_1_385964*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_384933�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_385967conv3d_2_385969*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_384950�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_385972conv3d_3_385974*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_384967�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_385977conv3d_4_385979*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_384984�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_385982conv3d_5_385984*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_385001�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_385987conv3d_6_385989*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_385018�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_385992conv3d_7_385994*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_385035�
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_385997conv3d_8_385999*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_385052�
'activity_regularization/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385406�
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
GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_384898�
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_9_386011conv3d_9_386013*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_385083�
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0conv3d_10_386016conv3d_10_386018*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_385100�
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_386021conv3d_11_386023*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_385117�
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0conv3d_12_386026conv3d_12_386028*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_385134�
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_386031conv3d_13_386033*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_385151�
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0conv3d_14_386036conv3d_14_386038*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_385168�
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0conv3d_15_386041conv3d_15_386043*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_385185�
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0conv3d_16_386046conv3d_16_386048*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_385202�
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_386051conv3d_17_386053*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_385218�
IdentityIdentity*conv3d_17/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�	
�
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_386791

inputs
identity

identity_1�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385406�
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
GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_384898l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������   S

Identity_1IdentityPartitionedCall_1:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_1_layer_call_fn_386616

inputs%
unknown:	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_384933{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_8_layer_call_fn_386756

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_385052{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_6_layer_call_and_return_conditional_losses_386727

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�q
�
A__inference_model_layer_call_and_return_conditional_losses_385226

inputs+
conv3d_384917:
conv3d_384919:-
conv3d_1_384934:	
conv3d_1_384936:	-
conv3d_2_384951:	
conv3d_2_384953:-
conv3d_3_384968:
conv3d_3_384970:-
conv3d_4_384985:
conv3d_4_384987:-
conv3d_5_385002:
conv3d_5_385004:-
conv3d_6_385019:
conv3d_6_385021:-
conv3d_7_385036:
conv3d_7_385038:-
conv3d_8_385053:
conv3d_8_385055:-
conv3d_9_385084:
conv3d_9_385086:.
conv3d_10_385101:
conv3d_10_385103:.
conv3d_11_385118:
conv3d_11_385120:.
conv3d_12_385135:
conv3d_12_385137:.
conv3d_13_385152:
conv3d_13_385154:.
conv3d_14_385169:
conv3d_14_385171:.
conv3d_15_385186:	
conv3d_15_385188:	.
conv3d_16_385203:	
conv3d_16_385205:.
conv3d_17_385219:
conv3d_17_385221:
identity

identity_1��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall�!conv3d_10/StatefulPartitionedCall�!conv3d_11/StatefulPartitionedCall�!conv3d_12/StatefulPartitionedCall�!conv3d_13/StatefulPartitionedCall�!conv3d_14/StatefulPartitionedCall�!conv3d_15/StatefulPartitionedCall�!conv3d_16/StatefulPartitionedCall�!conv3d_17/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall� conv3d_8/StatefulPartitionedCall� conv3d_9/StatefulPartitionedCall�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_384917conv3d_384919*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_384916�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_384934conv3d_1_384936*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_384933�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0conv3d_2_384951conv3d_2_384953*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_384950�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_384968conv3d_3_384970*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_384967�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0conv3d_4_384985conv3d_4_384987*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_4_layer_call_and_return_conditional_losses_384984�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_385002conv3d_5_385004*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_5_layer_call_and_return_conditional_losses_385001�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0conv3d_6_385019conv3d_6_385021*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_6_layer_call_and_return_conditional_losses_385018�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_385036conv3d_7_385038*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_385035�
 conv3d_8/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0conv3d_8_385053conv3d_8_385055*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_8_layer_call_and_return_conditional_losses_385052�
'activity_regularization/PartitionedCallPartitionedCall)conv3d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_activity_regularization_layer_call_and_return_conditional_losses_385062�
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
GPU2*0J 8� *H
fCRA
?__inference_activity_regularization_activity_regularizer_384898�
1activity_regularization/ActivityRegularizer/ShapeShape0activity_regularization/PartitionedCall:output:0*
T0*
_output_shapes
:�
?activity_regularization/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Aactivity_regularization/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
9activity_regularization/ActivityRegularizer/strided_sliceStridedSlice:activity_regularization/ActivityRegularizer/Shape:output:0Hactivity_regularization/ActivityRegularizer/strided_slice/stack:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_1:output:0Jactivity_regularization/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0activity_regularization/ActivityRegularizer/CastCastBactivity_regularization/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: �
3activity_regularization/ActivityRegularizer/truedivRealDivDactivity_regularization/ActivityRegularizer/PartitionedCall:output:04activity_regularization/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
 conv3d_9/StatefulPartitionedCallStatefulPartitionedCall0activity_regularization/PartitionedCall:output:0conv3d_9_385084conv3d_9_385086*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_9_layer_call_and_return_conditional_losses_385083�
!conv3d_10/StatefulPartitionedCallStatefulPartitionedCall)conv3d_9/StatefulPartitionedCall:output:0conv3d_10_385101conv3d_10_385103*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_10_layer_call_and_return_conditional_losses_385100�
!conv3d_11/StatefulPartitionedCallStatefulPartitionedCall*conv3d_10/StatefulPartitionedCall:output:0conv3d_11_385118conv3d_11_385120*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_385117�
!conv3d_12/StatefulPartitionedCallStatefulPartitionedCall*conv3d_11/StatefulPartitionedCall:output:0conv3d_12_385135conv3d_12_385137*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_385134�
!conv3d_13/StatefulPartitionedCallStatefulPartitionedCall*conv3d_12/StatefulPartitionedCall:output:0conv3d_13_385152conv3d_13_385154*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_13_layer_call_and_return_conditional_losses_385151�
!conv3d_14/StatefulPartitionedCallStatefulPartitionedCall*conv3d_13/StatefulPartitionedCall:output:0conv3d_14_385169conv3d_14_385171*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_385168�
!conv3d_15/StatefulPartitionedCallStatefulPartitionedCall*conv3d_14/StatefulPartitionedCall:output:0conv3d_15_385186conv3d_15_385188*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_15_layer_call_and_return_conditional_losses_385185�
!conv3d_16/StatefulPartitionedCallStatefulPartitionedCall*conv3d_15/StatefulPartitionedCall:output:0conv3d_16_385203conv3d_16_385205*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_385202�
!conv3d_17/StatefulPartitionedCallStatefulPartitionedCall*conv3d_16/StatefulPartitionedCall:output:0conv3d_17_385219conv3d_17_385221*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_17_layer_call_and_return_conditional_losses_385218�
IdentityIdentity*conv3d_17/StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   w

Identity_1Identity7activity_regularization/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall"^conv3d_10/StatefulPartitionedCall"^conv3d_11/StatefulPartitionedCall"^conv3d_12/StatefulPartitionedCall"^conv3d_13/StatefulPartitionedCall"^conv3d_14/StatefulPartitionedCall"^conv3d_15/StatefulPartitionedCall"^conv3d_16/StatefulPartitionedCall"^conv3d_17/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall!^conv3d_8/StatefulPartitionedCall!^conv3d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2F
!conv3d_10/StatefulPartitionedCall!conv3d_10/StatefulPartitionedCall2F
!conv3d_11/StatefulPartitionedCall!conv3d_11/StatefulPartitionedCall2F
!conv3d_12/StatefulPartitionedCall!conv3d_12/StatefulPartitionedCall2F
!conv3d_13/StatefulPartitionedCall!conv3d_13/StatefulPartitionedCall2F
!conv3d_14/StatefulPartitionedCall!conv3d_14/StatefulPartitionedCall2F
!conv3d_15/StatefulPartitionedCall!conv3d_15/StatefulPartitionedCall2F
!conv3d_16/StatefulPartitionedCall!conv3d_16/StatefulPartitionedCall2F
!conv3d_17/StatefulPartitionedCall!conv3d_17/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2D
 conv3d_8/StatefulPartitionedCall conv3d_8/StatefulPartitionedCall2D
 conv3d_9/StatefulPartitionedCall conv3d_9/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_3_layer_call_fn_386656

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_384967{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
*__inference_conv3d_16_layer_call_fn_386940

inputs%
unknown:	
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_16_layer_call_and_return_conditional_losses_385202{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�
�
E__inference_conv3d_14_layer_call_and_return_conditional_losses_385168

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�

&__inference_model_layer_call_fn_386221

inputs%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:	

unknown_30:	(

unknown_31:	

unknown_32:(

unknown_33:

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:���������   : *F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_385226{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_6_layer_call_and_return_conditional_losses_385018

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�

�
E__inference_conv3d_17_layer_call_and_return_conditional_losses_386970

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   k
IdentityIdentityBiasAdd:output:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
E__inference_conv3d_16_layer_call_and_return_conditional_losses_386951

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�
�
D__inference_conv3d_4_layer_call_and_return_conditional_losses_386687

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
E__inference_conv3d_16_layer_call_and_return_conditional_losses_385202

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�
�
E__inference_conv3d_12_layer_call_and_return_conditional_losses_386871

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
E__inference_conv3d_15_layer_call_and_return_conditional_losses_385185

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   	\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   	m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
o
S__inference_activity_regularization_layer_call_and_return_conditional_losses_386974

inputs
identityZ
IdentityIdentityinputs*
T0*3
_output_shapes!
:���������   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������   :[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
D__inference_conv3d_2_layer_call_and_return_conditional_losses_386647

inputs<
conv3d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:	*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   	
 
_user_specified_nameinputs
�
�
*__inference_conv3d_14_layer_call_fn_386900

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_14_layer_call_and_return_conditional_losses_385168{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
*__inference_conv3d_11_layer_call_fn_386840

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_11_layer_call_and_return_conditional_losses_385117{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�

&__inference_model_layer_call_fn_385850
input_1%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:	

unknown_30:	(

unknown_31:	

unknown_32:(

unknown_33:

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:���������   : *F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_385696{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
�
�
*__inference_conv3d_12_layer_call_fn_386860

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv3d_12_layer_call_and_return_conditional_losses_385134{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�

$__inference_signature_wrapper_386143
input_1%
unknown:
	unknown_0:'
	unknown_1:	
	unknown_2:	'
	unknown_3:	
	unknown_4:'
	unknown_5:
	unknown_6:'
	unknown_7:
	unknown_8:'
	unknown_9:

unknown_10:(

unknown_11:

unknown_12:(

unknown_13:

unknown_14:(

unknown_15:

unknown_16:(

unknown_17:

unknown_18:(

unknown_19:

unknown_20:(

unknown_21:

unknown_22:(

unknown_23:

unknown_24:(

unknown_25:

unknown_26:(

unknown_27:

unknown_28:(

unknown_29:	

unknown_30:	(

unknown_31:	

unknown_32:(

unknown_33:

unknown_34:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_384883{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������   : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������   
!
_user_specified_name	input_1
��
�K
"__inference__traced_restore_387701
file_prefix<
assignvariableop_conv3d_kernel:,
assignvariableop_1_conv3d_bias:@
"assignvariableop_2_conv3d_1_kernel:	.
 assignvariableop_3_conv3d_1_bias:	@
"assignvariableop_4_conv3d_2_kernel:	.
 assignvariableop_5_conv3d_2_bias:@
"assignvariableop_6_conv3d_3_kernel:.
 assignvariableop_7_conv3d_3_bias:@
"assignvariableop_8_conv3d_4_kernel:.
 assignvariableop_9_conv3d_4_bias:A
#assignvariableop_10_conv3d_5_kernel:/
!assignvariableop_11_conv3d_5_bias:A
#assignvariableop_12_conv3d_6_kernel:/
!assignvariableop_13_conv3d_6_bias:A
#assignvariableop_14_conv3d_7_kernel:/
!assignvariableop_15_conv3d_7_bias:A
#assignvariableop_16_conv3d_8_kernel:/
!assignvariableop_17_conv3d_8_bias:A
#assignvariableop_18_conv3d_9_kernel:/
!assignvariableop_19_conv3d_9_bias:B
$assignvariableop_20_conv3d_10_kernel:0
"assignvariableop_21_conv3d_10_bias:B
$assignvariableop_22_conv3d_11_kernel:0
"assignvariableop_23_conv3d_11_bias:B
$assignvariableop_24_conv3d_12_kernel:0
"assignvariableop_25_conv3d_12_bias:B
$assignvariableop_26_conv3d_13_kernel:0
"assignvariableop_27_conv3d_13_bias:B
$assignvariableop_28_conv3d_14_kernel:0
"assignvariableop_29_conv3d_14_bias:B
$assignvariableop_30_conv3d_15_kernel:	0
"assignvariableop_31_conv3d_15_bias:	B
$assignvariableop_32_conv3d_16_kernel:	0
"assignvariableop_33_conv3d_16_bias:B
$assignvariableop_34_conv3d_17_kernel:0
"assignvariableop_35_conv3d_17_bias:'
assignvariableop_36_adam_iter:	 )
assignvariableop_37_adam_beta_1: )
assignvariableop_38_adam_beta_2: (
assignvariableop_39_adam_decay: 0
&assignvariableop_40_adam_learning_rate: #
assignvariableop_41_total: #
assignvariableop_42_count: F
(assignvariableop_43_adam_conv3d_kernel_m:4
&assignvariableop_44_adam_conv3d_bias_m:H
*assignvariableop_45_adam_conv3d_1_kernel_m:	6
(assignvariableop_46_adam_conv3d_1_bias_m:	H
*assignvariableop_47_adam_conv3d_2_kernel_m:	6
(assignvariableop_48_adam_conv3d_2_bias_m:H
*assignvariableop_49_adam_conv3d_3_kernel_m:6
(assignvariableop_50_adam_conv3d_3_bias_m:H
*assignvariableop_51_adam_conv3d_4_kernel_m:6
(assignvariableop_52_adam_conv3d_4_bias_m:H
*assignvariableop_53_adam_conv3d_5_kernel_m:6
(assignvariableop_54_adam_conv3d_5_bias_m:H
*assignvariableop_55_adam_conv3d_6_kernel_m:6
(assignvariableop_56_adam_conv3d_6_bias_m:H
*assignvariableop_57_adam_conv3d_7_kernel_m:6
(assignvariableop_58_adam_conv3d_7_bias_m:H
*assignvariableop_59_adam_conv3d_8_kernel_m:6
(assignvariableop_60_adam_conv3d_8_bias_m:H
*assignvariableop_61_adam_conv3d_9_kernel_m:6
(assignvariableop_62_adam_conv3d_9_bias_m:I
+assignvariableop_63_adam_conv3d_10_kernel_m:7
)assignvariableop_64_adam_conv3d_10_bias_m:I
+assignvariableop_65_adam_conv3d_11_kernel_m:7
)assignvariableop_66_adam_conv3d_11_bias_m:I
+assignvariableop_67_adam_conv3d_12_kernel_m:7
)assignvariableop_68_adam_conv3d_12_bias_m:I
+assignvariableop_69_adam_conv3d_13_kernel_m:7
)assignvariableop_70_adam_conv3d_13_bias_m:I
+assignvariableop_71_adam_conv3d_14_kernel_m:7
)assignvariableop_72_adam_conv3d_14_bias_m:I
+assignvariableop_73_adam_conv3d_15_kernel_m:	7
)assignvariableop_74_adam_conv3d_15_bias_m:	I
+assignvariableop_75_adam_conv3d_16_kernel_m:	7
)assignvariableop_76_adam_conv3d_16_bias_m:I
+assignvariableop_77_adam_conv3d_17_kernel_m:7
)assignvariableop_78_adam_conv3d_17_bias_m:F
(assignvariableop_79_adam_conv3d_kernel_v:4
&assignvariableop_80_adam_conv3d_bias_v:H
*assignvariableop_81_adam_conv3d_1_kernel_v:	6
(assignvariableop_82_adam_conv3d_1_bias_v:	H
*assignvariableop_83_adam_conv3d_2_kernel_v:	6
(assignvariableop_84_adam_conv3d_2_bias_v:H
*assignvariableop_85_adam_conv3d_3_kernel_v:6
(assignvariableop_86_adam_conv3d_3_bias_v:H
*assignvariableop_87_adam_conv3d_4_kernel_v:6
(assignvariableop_88_adam_conv3d_4_bias_v:H
*assignvariableop_89_adam_conv3d_5_kernel_v:6
(assignvariableop_90_adam_conv3d_5_bias_v:H
*assignvariableop_91_adam_conv3d_6_kernel_v:6
(assignvariableop_92_adam_conv3d_6_bias_v:H
*assignvariableop_93_adam_conv3d_7_kernel_v:6
(assignvariableop_94_adam_conv3d_7_bias_v:H
*assignvariableop_95_adam_conv3d_8_kernel_v:6
(assignvariableop_96_adam_conv3d_8_bias_v:H
*assignvariableop_97_adam_conv3d_9_kernel_v:6
(assignvariableop_98_adam_conv3d_9_bias_v:I
+assignvariableop_99_adam_conv3d_10_kernel_v:8
*assignvariableop_100_adam_conv3d_10_bias_v:J
,assignvariableop_101_adam_conv3d_11_kernel_v:8
*assignvariableop_102_adam_conv3d_11_bias_v:J
,assignvariableop_103_adam_conv3d_12_kernel_v:8
*assignvariableop_104_adam_conv3d_12_bias_v:J
,assignvariableop_105_adam_conv3d_13_kernel_v:8
*assignvariableop_106_adam_conv3d_13_bias_v:J
,assignvariableop_107_adam_conv3d_14_kernel_v:8
*assignvariableop_108_adam_conv3d_14_bias_v:J
,assignvariableop_109_adam_conv3d_15_kernel_v:	8
*assignvariableop_110_adam_conv3d_15_bias_v:	J
,assignvariableop_111_adam_conv3d_16_kernel_v:	8
*assignvariableop_112_adam_conv3d_16_bias_v:J
,assignvariableop_113_adam_conv3d_17_kernel_v:8
*assignvariableop_114_adam_conv3d_17_bias_v:
identity_116��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�B
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*�A
value�AB�AtB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*�
value�B�tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypesx
v2t	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv3d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv3d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv3d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv3d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv3d_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv3d_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv3d_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv3d_8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv3d_9_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv3d_9_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv3d_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv3d_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv3d_11_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv3d_11_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv3d_12_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv3d_12_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv3d_13_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv3d_13_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv3d_14_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv3d_14_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv3d_15_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv3d_15_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv3d_16_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv3d_16_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv3d_17_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv3d_17_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_iterIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_beta_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_beta_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_decayIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_learning_rateIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_conv3d_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_conv3d_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv3d_1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv3d_1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv3d_2_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv3d_2_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv3d_3_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv3d_3_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv3d_4_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv3d_4_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv3d_5_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv3d_5_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv3d_6_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv3d_6_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv3d_7_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv3d_7_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv3d_8_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv3d_8_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv3d_9_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv3d_9_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv3d_10_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv3d_10_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv3d_11_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv3d_11_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv3d_12_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv3d_12_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv3d_13_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv3d_13_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv3d_14_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv3d_14_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv3d_15_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv3d_15_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv3d_16_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv3d_16_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv3d_17_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv3d_17_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adam_conv3d_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp&assignvariableop_80_adam_conv3d_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_conv3d_1_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_conv3d_1_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_conv3d_2_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_conv3d_2_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_conv3d_3_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_conv3d_3_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_conv3d_4_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_conv3d_4_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_conv3d_5_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_conv3d_5_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_conv3d_6_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_conv3d_6_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_conv3d_7_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_conv3d_7_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_conv3d_8_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_conv3d_8_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_conv3d_9_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_conv3d_9_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp+assignvariableop_99_adam_conv3d_10_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp*assignvariableop_100_adam_conv3d_10_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_conv3d_11_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_conv3d_11_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_conv3d_12_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_conv3d_12_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv3d_13_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv3d_13_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_conv3d_14_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_conv3d_14_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_conv3d_15_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_conv3d_15_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_conv3d_16_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_conv3d_16_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv3d_17_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv3d_17_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_115Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_116IdentityIdentity_115:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_116Identity_116:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142*
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
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
E__inference_conv3d_12_layer_call_and_return_conditional_losses_385134

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
)__inference_conv3d_7_layer_call_fn_386736

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv3d_7_layer_call_and_return_conditional_losses_385035{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs
�
�
E__inference_conv3d_13_layer_call_and_return_conditional_losses_385151

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������   \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������   m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������   w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������   
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
input_1<
serving_default_input_1:0���������   I
	conv3d_17<
StatefulPartitionedCall:0���������   tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
layer_with_weights-13
layer-15
layer_with_weights-14
layer-16
layer_with_weights-15
layer-17
layer_with_weights-16
layer-18
layer_with_weights-17
layer-19
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Kkernel
Lbias
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�!m�"m�'m�(m�-m�.m�3m�4m�9m�:m�?m�@m�Em�Fm�Km�Lm�Um�Vm�[m�\m�am�bm�gm�hm�mm�nm�sm�tm�ym�zm�m�	�m�	�m�	�m�v�v�!v�"v�'v�(v�-v�.v�3v�4v�9v�:v�?v�@v�Ev�Fv�Kv�Lv�Uv�Vv�[v�\v�av�bv�gv�hv�mv�nv�sv�tv�yv�zv�v�	�v�	�v�	�v�"
	optimizer
�
0
1
!2
"3
'4
(5
-6
.7
38
49
910
:11
?12
@13
E14
F15
K16
L17
U18
V19
[20
\21
a22
b23
g24
h25
m26
n27
s28
t29
y30
z31
32
�33
�34
�35"
trackable_list_wrapper
�
0
1
!2
"3
'4
(5
-6
.7
38
49
910
:11
?12
@13
E14
F15
K16
L17
U18
V19
[20
\21
a22
b23
g24
h25
m26
n27
s28
t29
y30
z31
32
�33
�34
�35"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
+:)2conv3d/kernel
:2conv3d/bias
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_1/kernel
:	2conv3d_1/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
#	variables
$trainable_variables
%regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+	2conv3d_2/kernel
:2conv3d_2/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_3/kernel
:2conv3d_3/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_4/kernel
:2conv3d_4/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_5/kernel
:2conv3d_5/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_6/kernel
:2conv3d_6/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_7/kernel
:2conv3d_7/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_8/kernel
:2conv3d_8/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
�__call__
�activity_regularizer_fn
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_9/kernel
:2conv3d_9/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_10/kernel
:2conv3d_10/bias
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_11/kernel
:2conv3d_11/bias
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_12/kernel
:2conv3d_12/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_13/kernel
:2conv3d_13/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_14/kernel
:2conv3d_14/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,	2conv3d_15/kernel
:	2conv3d_15/bias
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,	2conv3d_16/kernel
:2conv3d_16/bias
/
0
�1"
trackable_list_wrapper
/
0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2conv3d_17/kernel
:2conv3d_17/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
�
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
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
(
�0"
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
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
0:.2Adam/conv3d/kernel/m
:2Adam/conv3d/bias/m
2:0	2Adam/conv3d_1/kernel/m
 :	2Adam/conv3d_1/bias/m
2:0	2Adam/conv3d_2/kernel/m
 :2Adam/conv3d_2/bias/m
2:02Adam/conv3d_3/kernel/m
 :2Adam/conv3d_3/bias/m
2:02Adam/conv3d_4/kernel/m
 :2Adam/conv3d_4/bias/m
2:02Adam/conv3d_5/kernel/m
 :2Adam/conv3d_5/bias/m
2:02Adam/conv3d_6/kernel/m
 :2Adam/conv3d_6/bias/m
2:02Adam/conv3d_7/kernel/m
 :2Adam/conv3d_7/bias/m
2:02Adam/conv3d_8/kernel/m
 :2Adam/conv3d_8/bias/m
2:02Adam/conv3d_9/kernel/m
 :2Adam/conv3d_9/bias/m
3:12Adam/conv3d_10/kernel/m
!:2Adam/conv3d_10/bias/m
3:12Adam/conv3d_11/kernel/m
!:2Adam/conv3d_11/bias/m
3:12Adam/conv3d_12/kernel/m
!:2Adam/conv3d_12/bias/m
3:12Adam/conv3d_13/kernel/m
!:2Adam/conv3d_13/bias/m
3:12Adam/conv3d_14/kernel/m
!:2Adam/conv3d_14/bias/m
3:1	2Adam/conv3d_15/kernel/m
!:	2Adam/conv3d_15/bias/m
3:1	2Adam/conv3d_16/kernel/m
!:2Adam/conv3d_16/bias/m
3:12Adam/conv3d_17/kernel/m
!:2Adam/conv3d_17/bias/m
0:.2Adam/conv3d/kernel/v
:2Adam/conv3d/bias/v
2:0	2Adam/conv3d_1/kernel/v
 :	2Adam/conv3d_1/bias/v
2:0	2Adam/conv3d_2/kernel/v
 :2Adam/conv3d_2/bias/v
2:02Adam/conv3d_3/kernel/v
 :2Adam/conv3d_3/bias/v
2:02Adam/conv3d_4/kernel/v
 :2Adam/conv3d_4/bias/v
2:02Adam/conv3d_5/kernel/v
 :2Adam/conv3d_5/bias/v
2:02Adam/conv3d_6/kernel/v
 :2Adam/conv3d_6/bias/v
2:02Adam/conv3d_7/kernel/v
 :2Adam/conv3d_7/bias/v
2:02Adam/conv3d_8/kernel/v
 :2Adam/conv3d_8/bias/v
2:02Adam/conv3d_9/kernel/v
 :2Adam/conv3d_9/bias/v
3:12Adam/conv3d_10/kernel/v
!:2Adam/conv3d_10/bias/v
3:12Adam/conv3d_11/kernel/v
!:2Adam/conv3d_11/bias/v
3:12Adam/conv3d_12/kernel/v
!:2Adam/conv3d_12/bias/v
3:12Adam/conv3d_13/kernel/v
!:2Adam/conv3d_13/bias/v
3:12Adam/conv3d_14/kernel/v
!:2Adam/conv3d_14/bias/v
3:1	2Adam/conv3d_15/kernel/v
!:	2Adam/conv3d_15/bias/v
3:1	2Adam/conv3d_16/kernel/v
!:2Adam/conv3d_16/bias/v
3:12Adam/conv3d_17/kernel/v
!:2Adam/conv3d_17/bias/v
�2�
&__inference_model_layer_call_fn_385302
&__inference_model_layer_call_fn_386221
&__inference_model_layer_call_fn_386299
&__inference_model_layer_call_fn_385850�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_model_layer_call_and_return_conditional_losses_386443
A__inference_model_layer_call_and_return_conditional_losses_386587
A__inference_model_layer_call_and_return_conditional_losses_385954
A__inference_model_layer_call_and_return_conditional_losses_386058�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
!__inference__wrapped_model_384883input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv3d_layer_call_fn_386596�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv3d_layer_call_and_return_conditional_losses_386607�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_1_layer_call_fn_386616�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_1_layer_call_and_return_conditional_losses_386627�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_2_layer_call_fn_386636�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_2_layer_call_and_return_conditional_losses_386647�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_3_layer_call_fn_386656�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_3_layer_call_and_return_conditional_losses_386667�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_4_layer_call_fn_386676�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_4_layer_call_and_return_conditional_losses_386687�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_5_layer_call_fn_386696�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_5_layer_call_and_return_conditional_losses_386707�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_6_layer_call_fn_386716�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_6_layer_call_and_return_conditional_losses_386727�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_7_layer_call_fn_386736�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_7_layer_call_and_return_conditional_losses_386747�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv3d_8_layer_call_fn_386756�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_8_layer_call_and_return_conditional_losses_386767�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_activity_regularization_layer_call_fn_386772
8__inference_activity_regularization_layer_call_fn_386777�
���
FullArgSpec
args�
jself
jinputs
varargsjargs
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_386784
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_386791�
���
FullArgSpec
args�
jself
jinputs
varargsjargs
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
)__inference_conv3d_9_layer_call_fn_386800�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv3d_9_layer_call_and_return_conditional_losses_386811�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv3d_10_layer_call_fn_386820�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv3d_10_layer_call_and_return_conditional_losses_386831�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv3d_11_layer_call_fn_386840�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv3d_11_layer_call_and_return_conditional_losses_386851�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv3d_12_layer_call_fn_386860�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv3d_12_layer_call_and_return_conditional_losses_386871�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv3d_13_layer_call_fn_386880�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv3d_13_layer_call_and_return_conditional_losses_386891�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv3d_14_layer_call_fn_386900�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv3d_14_layer_call_and_return_conditional_losses_386911�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv3d_15_layer_call_fn_386920�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv3d_15_layer_call_and_return_conditional_losses_386931�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv3d_16_layer_call_fn_386940�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv3d_16_layer_call_and_return_conditional_losses_386951�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv3d_17_layer_call_fn_386960�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv3d_17_layer_call_and_return_conditional_losses_386970�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_386143input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_activity_regularization_activity_regularizer_384898�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�2�
S__inference_activity_regularization_layer_call_and_return_conditional_losses_386974
S__inference_activity_regularization_layer_call_and_return_conditional_losses_386978�
���
FullArgSpec
args�
jself
jinputs
varargsjargs
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
!__inference__wrapped_model_384883�'!"'(-.349:?@EFKLUV[\abghmnstyz���<�9
2�/
-�*
input_1���������   
� "A�>
<
	conv3d_17/�,
	conv3d_17���������   i
?__inference_activity_regularization_activity_regularizer_384898&�
�
�	
x
� "� �
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_386784�K�H
1�.
,�)
inputs���������   
�

trainingp "?�<
'�$
0���������   
�
�	
1/0 �
W__inference_activity_regularization_layer_call_and_return_all_conditional_losses_386791�K�H
1�.
,�)
inputs���������   
�

trainingp"?�<
'�$
0���������   
�
�	
1/0 �
S__inference_activity_regularization_layer_call_and_return_conditional_losses_386974�K�H
1�.
,�)
inputs���������   
�

trainingp "1�.
'�$
0���������   
� �
S__inference_activity_regularization_layer_call_and_return_conditional_losses_386978�K�H
1�.
,�)
inputs���������   
�

trainingp"1�.
'�$
0���������   
� �
8__inference_activity_regularization_layer_call_fn_386772sK�H
1�.
,�)
inputs���������   
�

trainingp "$�!���������   �
8__inference_activity_regularization_layer_call_fn_386777sK�H
1�.
,�)
inputs���������   
�

trainingp"$�!���������   �
E__inference_conv3d_10_layer_call_and_return_conditional_losses_386831t[\;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
*__inference_conv3d_10_layer_call_fn_386820g[\;�8
1�.
,�)
inputs���������   
� "$�!���������   �
E__inference_conv3d_11_layer_call_and_return_conditional_losses_386851tab;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
*__inference_conv3d_11_layer_call_fn_386840gab;�8
1�.
,�)
inputs���������   
� "$�!���������   �
E__inference_conv3d_12_layer_call_and_return_conditional_losses_386871tgh;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
*__inference_conv3d_12_layer_call_fn_386860ggh;�8
1�.
,�)
inputs���������   
� "$�!���������   �
E__inference_conv3d_13_layer_call_and_return_conditional_losses_386891tmn;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
*__inference_conv3d_13_layer_call_fn_386880gmn;�8
1�.
,�)
inputs���������   
� "$�!���������   �
E__inference_conv3d_14_layer_call_and_return_conditional_losses_386911tst;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
*__inference_conv3d_14_layer_call_fn_386900gst;�8
1�.
,�)
inputs���������   
� "$�!���������   �
E__inference_conv3d_15_layer_call_and_return_conditional_losses_386931tyz;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   	
� �
*__inference_conv3d_15_layer_call_fn_386920gyz;�8
1�.
,�)
inputs���������   
� "$�!���������   	�
E__inference_conv3d_16_layer_call_and_return_conditional_losses_386951u�;�8
1�.
,�)
inputs���������   	
� "1�.
'�$
0���������   
� �
*__inference_conv3d_16_layer_call_fn_386940h�;�8
1�.
,�)
inputs���������   	
� "$�!���������   �
E__inference_conv3d_17_layer_call_and_return_conditional_losses_386970v��;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
*__inference_conv3d_17_layer_call_fn_386960i��;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_1_layer_call_and_return_conditional_losses_386627t!";�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   	
� �
)__inference_conv3d_1_layer_call_fn_386616g!";�8
1�.
,�)
inputs���������   
� "$�!���������   	�
D__inference_conv3d_2_layer_call_and_return_conditional_losses_386647t'(;�8
1�.
,�)
inputs���������   	
� "1�.
'�$
0���������   
� �
)__inference_conv3d_2_layer_call_fn_386636g'(;�8
1�.
,�)
inputs���������   	
� "$�!���������   �
D__inference_conv3d_3_layer_call_and_return_conditional_losses_386667t-.;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_3_layer_call_fn_386656g-.;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_4_layer_call_and_return_conditional_losses_386687t34;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_4_layer_call_fn_386676g34;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_5_layer_call_and_return_conditional_losses_386707t9:;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_5_layer_call_fn_386696g9:;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_6_layer_call_and_return_conditional_losses_386727t?@;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_6_layer_call_fn_386716g?@;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_7_layer_call_and_return_conditional_losses_386747tEF;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_7_layer_call_fn_386736gEF;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_8_layer_call_and_return_conditional_losses_386767tKL;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_8_layer_call_fn_386756gKL;�8
1�.
,�)
inputs���������   
� "$�!���������   �
D__inference_conv3d_9_layer_call_and_return_conditional_losses_386811tUV;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
)__inference_conv3d_9_layer_call_fn_386800gUV;�8
1�.
,�)
inputs���������   
� "$�!���������   �
B__inference_conv3d_layer_call_and_return_conditional_losses_386607t;�8
1�.
,�)
inputs���������   
� "1�.
'�$
0���������   
� �
'__inference_conv3d_layer_call_fn_386596g;�8
1�.
,�)
inputs���������   
� "$�!���������   �
A__inference_model_layer_call_and_return_conditional_losses_385954�'!"'(-.349:?@EFKLUV[\abghmnstyz���D�A
:�7
-�*
input_1���������   
p 

 
� "?�<
'�$
0���������   
�
�	
1/0 �
A__inference_model_layer_call_and_return_conditional_losses_386058�'!"'(-.349:?@EFKLUV[\abghmnstyz���D�A
:�7
-�*
input_1���������   
p

 
� "?�<
'�$
0���������   
�
�	
1/0 �
A__inference_model_layer_call_and_return_conditional_losses_386443�'!"'(-.349:?@EFKLUV[\abghmnstyz���C�@
9�6
,�)
inputs���������   
p 

 
� "?�<
'�$
0���������   
�
�	
1/0 �
A__inference_model_layer_call_and_return_conditional_losses_386587�'!"'(-.349:?@EFKLUV[\abghmnstyz���C�@
9�6
,�)
inputs���������   
p

 
� "?�<
'�$
0���������   
�
�	
1/0 �
&__inference_model_layer_call_fn_385302�'!"'(-.349:?@EFKLUV[\abghmnstyz���D�A
:�7
-�*
input_1���������   
p 

 
� "$�!���������   �
&__inference_model_layer_call_fn_385850�'!"'(-.349:?@EFKLUV[\abghmnstyz���D�A
:�7
-�*
input_1���������   
p

 
� "$�!���������   �
&__inference_model_layer_call_fn_386221�'!"'(-.349:?@EFKLUV[\abghmnstyz���C�@
9�6
,�)
inputs���������   
p 

 
� "$�!���������   �
&__inference_model_layer_call_fn_386299�'!"'(-.349:?@EFKLUV[\abghmnstyz���C�@
9�6
,�)
inputs���������   
p

 
� "$�!���������   �
$__inference_signature_wrapper_386143�'!"'(-.349:?@EFKLUV[\abghmnstyz���G�D
� 
=�:
8
input_1-�*
input_1���������   "A�>
<
	conv3d_17/�,
	conv3d_17���������   