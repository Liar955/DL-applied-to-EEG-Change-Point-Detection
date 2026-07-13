%% ========= EDF to 200-500 Hz to H5 (SAFE, NO DOWNSAMPLING) =========
% Goal: robust band-pass preprocessing, avoid false change points, normalize warnings, and avoid logical H5 attributes.
% Notes:
%  - If Fs is below 1000 Hz, the 500 Hz high cutoff is clipped to Nyquist*0.99.
%  - Bad channels are detected once globally and kept consistent across blocks.
%  - If filtfilt is not feasible, fall back to one-pass filter and count those blocks.
%  - Do not linearly interpolate; set NaN/Inf to zero only. Mask writing is optional.
%  - H5 attributes are written as numeric values or strings only, not logical values.
%  - All warnings use identifiers and formatted strings.

%% Parameters
rootDir   = 'path/to/raw_edf_files';
fileType  = '*.edf';
filterOrd = 4;              % Butterworth filter order
bandWant  = [200, 500];     % Target band (Hz)
minDur    = 1;              % Minimum duration (s); warn only if shorter
blk       = 2e5;            % Block size
ovl       = 2e3;            % Block overlap

%% Main loop
files = dir(fullfile(rootDir,'**',fileType));
fprintf('Found %d files under %s\n\n', numel(files), rootDir);

for k = 1:numel(files)
    fname  = files(k).name;
    fpath  = fullfile(files(k).folder, fname);
    [~,stem] = fileparts(fname);
    outFile = fullfile(files(k).folder, [stem '_filt.h5']);
    tmpFile = fullfile(files(k).folder, [stem '_filt.tmp.h5']);
    fprintf('[%3d/%d] %s\n', k, numel(files), fname);

    try
        % Read and validate without downsampling.
        [sig, Fs] = read_and_validate_safe(fpath, bandWant, minDur, ...
                                           'MinChan', 31, 'StrictFs', false);

        % Design a safe band-pass filter; clip high cutoff if Fs is insufficient; fall back to low-pass if needed.
        [sos, g, band_used, mode] = design_bp_safe(filterOrd, bandWant, Fs);

        % Filter in blocks and save safely.
        if exist(tmpFile,'file'), delete(tmpFile); end
        stats = block_filter_and_save_safe(sig, sos, g, Fs, tmpFile, blk, ovl, ...
                                           'WriteMask', false); % Set true to write a mask if needed

        % Atomically move the completed result.
        if exist(outFile,'file'), delete(outFile); end
        movefile(tmpFile, outFile);

        % Write metadata; avoid logical attributes.
        try
            h5writeatt(outFile,'/sig','Fs',Fs);
            h5writeatt(outFile,'/sig','Fs_original',Fs);
            h5writeatt(outFile,'/sig','band_nominal',bandWant);
            h5writeatt(outFile,'/sig','band_used',band_used);
            h5writeatt(outFile,'/sig','design_mode', char(mode));         % char/string
            h5writeatt(outFile,'/sig','blk',blk);
            h5writeatt(outFile,'/sig','ovl',ovl);
            h5writeatt(outFile,'/sig','bad_channel_count', int32(stats.nBad));
            h5writeatt(outFile,'/sig','used_filtfilt_blocks', int32(stats.nFF));
            h5writeatt(outFile,'/sig','fallback_filter_blocks', int32(stats.nF1));
            h5writeatt(outFile,'/sig','downsample', int32(0));             % 0=no downsampling
            h5writeatt(outFile,'/sig','safe_mode', int32(1));
        catch MEatt
            warning('xf:attwrite','%s', MEatt.message);
        end

        fprintf('   Saved → %s\n\n', outFile);

    catch ME
        if exist(tmpFile,'file'), delete(tmpFile); end
        warning('xf:processFailed','%s failed: %s', fname, ME.message);
    end
end


%% Function definitions

function [sig,Fs,meta] = read_and_validate_safe(fpath, band, minDur, varargin)
% Robust EDF reading; avoid incorrect reads; no downsampling.
% Options: 'MinChan' (default 1), 'StrictFs' (default false)
p = inputParser;
addParameter(p,'MinChan',1);
addParameter(p,'StrictFs',false);
parse(p,varargin{:});
minChan  = p.Results.MinChan;
strictFs = p.Results.StrictFs;

hdr = edfinfo(fpath);
raw = edfread(fpath);
if istable(raw) || istimetable(raw)
    sig = raw.Variables;
else
    sig = raw;
end
if iscell(sig), sig = cell2mat(sig); end
sig = double(sig);

Fs  = hdr.NumSamples(1) / seconds(hdr.DataRecordDuration);
if any(hdr.NumSamples ~= hdr.NumSamples(1))
    warning('xf:edf','channels sample-count differ; use ch1 Fs=%.2f Hz', Fs);
end

if ndims(sig)==3
    [N,C,S] = size(sig); %#ok<NASGU>
    sig = reshape(permute(sig,[1 3 2]), [], size(sig,3));
end

[nSamp, nChan] = size(sig);
if nChan < minChan
    error('channels=%d (<%d required)', nChan, minChan);
end
if nSamp < Fs*minDur
    warning('xf:shortDur','duration %.2fs < %.2fs; proceed anyway', nSamp/Fs, minDur);
end

if strictFs
    if Fs < 2*band(2)
        error('Fs=%.1f < 2×%.0f (need >=1000 Hz for 500 Hz highcut)', Fs, band(2));
    end
else
    if Fs < 2*band(1)
        error('Fs=%.1f < 2×%.0f (need >=400 Hz for 200 Hz lowcut)', Fs, band(1));
    end
end

meta = struct();
end


function [sos,g,band_used,mode] = design_bp_safe(ord, bandWant, Fs)
% Safely design a band-pass filter; clip high cutoff if Fs is insufficient; fall back to low-pass if lo >= hi.
hi = min(bandWant(2), Fs/2*0.99);
band_used = [bandWant(1) hi];
mode = 'exact';

if hi < bandWant(2)
    mode = 'clipped';  % Fs is insufficient for a 500 Hz high cutoff
end
if bandWant(1) >= hi
    % Band-pass cannot be realized; fall back to low-pass while preserving high-frequency information as much as possible.
    [b,a] = butter(ord, hi/(Fs/2), 'low');
    [sos,g] = tf2sos(b,a);
    band_used = [0 hi];
    mode = 'lowpass_fallback';
    return;
end

[b,a] = butter(ord, [bandWant(1) hi]/(Fs/2), 'bandpass');
[sos,g] = tf2sos(b,a);
end


function stats = block_filter_and_save_safe(sig, sos, g, Fs, outFile, blk, ovl, varargin)
% Safe block-wise filtering and H5 writing:
%  - Detect bad channels once globally and keep behavior consistent across blocks.
%  - Do not linearly interpolate; zero-fill only.
%  - Fall back to one-pass filter if filtfilt is not feasible.
% Option: 'WriteMask' controls whether invalid-sample mask /mask is written.
p = inputParser;
addParameter(p,'WriteMask',false);
parse(p,varargin{:});
writeMask = p.Results.WriteMask;

N = size(sig,1); 
C = size(sig,2);

% (1) Global bad channels: NaN/Inf or zero-variance channels.
col_bad = any(~isfinite(sig),1) | std(sig,'omitnan')==0;
nBad = sum(col_bad);
if nBad>0
    fprintf('   drop %d bad channel(s) -> zero-fill\n', nBad);
end
sig(:,col_bad) = 0;  % Zero-fill consistently while preserving dimensions

% (2) Adaptive pad length to ensure sufficient overlap.
padlen_est = 3*max(8, 2*size(sos,1));
if ovl < padlen_est
    fprintf('   ovl auto increase from %d to %d (pad need)\n', ovl, padlen_est);
    ovl = padlen_est;
end

% (3) Create H5 output.
h5create(outFile, '/sig', [Inf C], 'Datatype','single', ...
         'ChunkSize', [min(blk,N) C], 'Deflate',5);
h5writeatt(outFile,'/sig','Fs',Fs);
h5writeatt(outFile,'/sig','safe_mode',int32(1));
h5writeatt(outFile,'/sig','bad_channel_count',int32(nBad));

% Optionally write mask: 1=valid, 0=invalid.
if writeMask
    mask = isfinite(sig); % Convert logical values to uint8 below
    h5create(outFile,'/mask',[Inf C],'Datatype','uint8',...
             'ChunkSize', [min(blk,N) C], 'Deflate',5);
    h5writeatt(outFile,'/mask','desc','1=finite, 0=non-finite-before-clean');
end

nFF = 0; nF1 = 0;  % Count filtfilt and one-pass filter blocks

% (4) Block processing.
for i0 = 1:blk:N
    i1  = min(N, i0+blk-1);
    i0e = max(1, i0-ovl);
    i1e = min(N, i1+ovl);
    tmp = sig(i0e:i1e, :);

    % Do not interpolate; only set non-finite values to zero.
    tmp(~isfinite(tmp)) = 0;

    % Check whether filtfilt is feasible.
    useFF = (size(tmp,1) > padlen_est+2);
    if useFF
        seg = filtfilt(sos, g, tmp); nFF = nFF + 1;
    else
        warning('xf:shortSeg','short seg %d–%d -> use one-pass filter()', i0, i1);
        seg = filter(sos, g, tmp);   nF1 = nF1 + 1;
    end

    keep = (i0:i1) - (i0e-1);
    h5write(outFile, '/sig', single(seg(keep,:)), [i0 1], size(seg(keep,:)));

    if writeMask
        mseg = uint8(isfinite(sig(i0:i1,:))); % Convert to uint8
        h5write(outFile, '/mask', mseg, [i0 1], size(mseg));
    end

    fprintf('    %d–%d\n', i0, i1);
end

stats = struct('nBad', nBad, 'nFF', nFF, 'nF1', nF1);
end
