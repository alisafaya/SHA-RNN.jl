# source: https://gist.github.com/denizyuret/bf74f737ba8a7076a7f43c5080555a9a

using AutoGrad, CuArrays, Knet, Test
using AutoGrad: @gcheck
using CuArrays: @cufunc

const GConstant01 = sqrt(2/pi)
const GConstant02 = 0.044715 * sqrt(2/pi)
const GConstant03 = GConstant01 / 2

# Main definition, broadcasted version works on Arrays

gelu(x::T) where T = (x/2)*(1 + tanh(T(GConstant02)*x^3 + T(GConstant01)*x))
geluback(x::T,dy::T) where T = dy*(T(0.5)*tanh(T(GConstant02)*x^3 + T(GConstant01)*x) + (T(0.0535161)*x^3 + T(GConstant03)*x)*(1/cosh(T(GConstant02)*x^3 + T(GConstant01)*x))^2 + T(0.5))


# This defines gelu for AutoGrad

@primitive  gelu(x),dy  geluback.(x,dy)


# This makes broadcasted gelu work for CuArray, directly compiling a CUDA kernel from Julia code.
# @cufunc necessary to use a GPU compatible version of tanh etc.
# https://github.com/JuliaGPU/CuArrays.jl/issues/253

@cufunc gelu(x::T) where T = (x/2)*(1 + tanh(T(GConstant02)*x^3 + T(GConstant01)*x))
@cufunc geluback(x::T,dy::T) where T = dy*(T(0.5)*tanh(T(GConstant02)*x^3 + T(GConstant01)*x) + (T(0.0535161)*x^3 + T(GConstant03)*x)*(1/cosh(T(GConstant02)*x^3 + T(GConstant01)*x))^2 + T(0.5))


# This defines gelu for KnetArray
import Base.Broadcast: broadcasted
import Knet: KnetArray

function KnetArray(x::CuArray{T,N}) where {T,N}
    p = Base.bitcast(Knet.Cptr, x.ptr)
    k = Knet.KnetPtr(p, sizeof(x), gpu(), x) 
    KnetArray{T,N}(k, size(x))
end

broadcasted(::typeof(gelu),x::KnetArray) = KnetArray(gelu.(CuArray(x)))
broadcasted(::typeof(geluback),x::KnetArray,dy::KnetArray) = KnetArray(geluback.(CuArray(x),CuArray(dy)))


# This tests all of the above
# julia -L gelu.jl -e 'testgelu()'
function testgelu()
    @testset "gelu" begin
        x = rand(10)
        a = Param(x)
        c = Param(CuArray(x))
        k = Param(KnetArray(x))
        @test @gcheck gelu.(a)
        @test @gcheck gelu.(c)
        @test @gcheck gelu.(k)
    end
end

"""
    gelu(x)
    
    Gaussian Error Linear Unit Activation Function
    https://arxiv.org/abs/1606.08415

    returns: 
        0.5x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))
"""
gelu