function [ initParam ] = InitAdamParamEx(sz)
initParam.m = gpuArray.zeros(sz);
initParam.v = gpuArray.zeros(sz);
initParam.beta1pow = gpuArray.ones(sz);
initParam.beta2pow = gpuArray.ones(sz);
end

