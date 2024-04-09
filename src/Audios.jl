module Audios

using WAV, DSP
using StatsBase
using ProgressMeter

export import_audios,
	make_spectrograms,
	pad_spectrograms

# read WAV files into an array
function import_audios(files::Vector{String})
	signal = []
	sr = []
	@showprogress "Importing audios..." for f ∈ files
		y, s = wavread(f)
		push!(signal, vec(y))
		push!(sr, s)
	end
	return signal, sr
end

# spectrograms
mutable struct Spectrogram
	time::Vector{Float64}
	frequency::Vector{Float64}
	db::Matrix{Float64}

	function Spectrogram(signal::Vector{Float64})
		spec = spectrogram(signal, 509)
		return new(spec.time, log.(spec.freq), pow2db.(spec.power))
	end
end

# input data
function make_spectrograms(x::Vector; trans = nothing)
	@showprogress "Generating spectrograms..." for (i, xi) ∈ enumerate(x)
		xi = transpose(Spectrogram(xi).db)
		if trans !== nothing
			dt = fit(trans, xi, dims = 2)
			xi = StatsBase.transform(dt, xi)
		end
		x[i] = xi
	end
	return x
end

# pad zeroes to make all matrices of same size
function pad_spectrograms(x::Vector, cols::Int)
	@showprogress "Padding spectograms to $(cols) time steps..." for (i, xi) ∈ enumerate(x)
		if size(xi, 1) < cols
			z = rand(Float64, (cols - size(xi, 1), size(xi, 2)))
			xi = vcat(xi, z)
		elseif size(xi, 1) > cols
			@warn "Element $(i) more than $(cols) timestamps"
		end
		x[i] = xi
	end
	return x
	@info "Padded spectograms to be $(cols) time steps long"
end

end
