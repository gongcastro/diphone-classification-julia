module Audios

using WAV, DSP
using StatsBase
using ProgressMeter

export import_audio,
	make_spectrograms,
	make_data

# read WAV files into an array
function import_audio(files::Vector{String})
	signal = []
	sr = []
	@showprogress for f ∈ files
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

function make_spectrograms(x::Vector)
	spec_list = Spectrogram[]
	@showprogress for s ∈ x
		spec = Spectrogram(s)
		push!(spec_list, spec)
	end
	return spec_list
end


# input data
function make_data(x::Vector; pad::Bool = true, trans = nothing)

	max_cols = maximum(map(x -> size(x.db, 2), x))
	y = rand(max_cols, size(first(x).db, 1), length(x))

	@showprogress for (i, s) ∈ enumerate(x)
		y_i = transpose(s.db)
		dim = size(y_i)
		# pad zeroes to make all matrices of same size
		if pad && dim[1] < max_cols
			z = rand(Float64, (max_cols - dim[1], dim[2]))
			y_i = vcat(y_i, z)
		end

		if trans ≠ nothing
			dt = fit(trans, y_i, dims = 2)
			y_i = StatsBase.transform(dt, y_i)
		end
		y[:, :, i] = y_i
	end
	return y
end

end
