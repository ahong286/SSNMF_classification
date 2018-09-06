function compile_mex(compiler_type)

opts = { '-I.', '-O', '-v', '-largeArrayDims'} ;

if nargin < 1
    try
        cc = mex.getCompilerConfigurations('C').Name;
    catch
        cc ='';
        compiler_type='unknown';
    end
    if ~isempty(strfind(lower(cc),lower('Microsoft Visual C++')));
        compiler_type = 'visualc' ;
    elseif ~isempty(strfind(lower(cc),lower('Intel C++')));
        compiler_type = 'intelc' ;
    elseif ~isempty(strfind(lower(cc),lower('GNU C')));
        compiler_type = 'gcc' ;
    end
    if ~exist('compiler_type','var')
        compiler_type='unknown';
    end
end


switch computer
    case {'PCWIN', 'PCWIN64'}
        opts = [opts, {'OPTIMFLAGS=$OPTIMFLAGS /openmp'}] ;
        if strcmpi(compiler_type,'visualc')
            opts = [opts, {'COMPFLAGS=$COMPFLAGS /MT /TC'}] ;
            opts = [opts, {'OPTIMFLAGS=$OPTIMFLAGS /O3 /Oi /Ot /Oy /DNDEBUG /fp:fast'}] ;
        elseif strcmpi(compiler_type,'intelc')
            opts = [opts, {'COMPFLAGS=$COMPFLAGS /MT /TC'}] ;
            opts = [opts, {'OPTIMFLAGS=$OPTIMFLAGS /O3 /Ot /Oi /DNDEBUG /fp:fast=2 /Qfp-speculation:fast /Qparallel /Qprec-div- /Qprec-sqrt- /Qfma /Qfast-transcendentals'}];
        end
        
    case {'MAC', 'MACI', 'MACI64','GLNX86', 'GLNXA64'}
            opts = [opts, {'COPTIMFLAGS=\$COPTIMFLAGS -fopenmp'}] ;
            opts = [opts, {'LDOPTIMFLAGS=\$LDOPTIMFLAGS -fopenmp'}] ;
        if strcmpi(compiler_type,'gcc')
            opts = [opts, {'CFLAGS=\$CFLAGS -x c'}];
            opts = [opts, {'COPTIMFLAGS=\$COPTIMFLAGS -Ofast -fomit-frame-pointer -ffast-math -fprefetch-loop-arrays'}];
            opts = [opts, {'LDOPTIMFLAGS=\$LDOPTIMFLAGS -Ofast'}];
        end
end

try
    sift_compile_work(opts);
catch
    warning('The C files are not successfully compiled, trying to compile again with no optimize options.');
    opts = { '-I.', '-O', '-v', '-largeArrayDims'} ;
    sift_compile_work(opts);
end

function sift_compile_work(opts)
mex('col2imstep.c',opts{:}) ;
mex('countcover.c',opts{:}) ;
mex('im2colstep.c',opts{:}) ;

