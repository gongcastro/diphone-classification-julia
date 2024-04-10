module Model

include("./Audios.jl")
using .Audios
using Flux, CUDA, Statistics, StatsBase, OneHotArrays
using Flux.Losses: mse
using ProgressMeter

export make_features,
	make_targets,
	Args,
	get_loss,
	get_accuracy,
	make_predictions,
	train_model

function make_features(files::Vector{String}; pad::Bool = true)
	audios, sr = import_audios(files)
	x = make_spectrograms(audios; trans = StatsBase.UnitRangeTransform)
	max_cols = maximum(map(x -> size(x, 1), x))
	if pad
		x = pad_spectrograms(x, max_cols)
	end
	x = convert.(Array{Float32}, x)
	return transpose.(x)
end

function make_targets(files::Vector{String})
	labels = getindex.(split.(basename.(files), "_"), 1)
	targets = convert.(Int32, labels .== "CV")
	return targets, labels
end


Base.@kwdef mutable struct Args
	batch_size::Int = 64 # batch size
	η::Float64 = 1e-3 # learning rate
	epochs::Int = 1e3 # number of epochs
	throttle::Int = 10
end


function get_accuracy(result, label)
	pred = convert.(Int, result .>= 0.5)
	acc = mean(pred .== label)
	return acc
end

function make_predictions(model, x, y)
	probs = [model(i) for i in x]
	preds = [p .>= 0.5 for p in probs]
	last_pred = [x[end] for (x) in preds]
	acc = mean(last_pred .== y)

	return probs, preds, acc
end

end


