begin
	include("./Audios.jl")
	using .Audios
	using Flux, CUDA, Statistics, StatsBase, OneHotArrays
	using Plots, ProgressMeter
	Plots.PythonPlotBackend()
end

# list WAV files
audio_path = joinpath("sounds", "raw")
wav_files = readdir(audio_path, join = true)[contains.(lowercase.(readdir(audio_path)), ".wav")]
wav_names = basename.(wav_files)
N = length(wav_names)
# import and preprocess data
audios, sr = import_audio(wav_files)
specs = make_spectrograms(audios)
data = make_data(specs, pad = true; trans = StatsBase.UnitRangeTransform)
data = convert(Array{Float32}, data)


X = []
for i ∈ collect(range(1, size(data, 3)))
	push!(X, transpose(data[:, :, i]))
end

# get labels
labels = getindex.(split.(wav_names, "_"), 1)
const classes = unique(labels)
targets = labels .== "CV"

# Model -----------------------------
N = length(data)
n_input = size(X[1], 1)
n_hidden = trunc(Int, n_input / 2)
n_output = length(unique(targets)) - 1

# model structure
model = Chain(
	RNN(n_input => n_hidden, tanh),
	Dense(n_hidden => n_output),
	sigmoid,
)

n_epochs = 100
opt = ADAM(0.001)
θ = Flux.params(model) # Keep track of the model parameters

@showprogress for epoch ∈ 1:n_epochs # Training loop
	Flux.reset!(model) # Reset the hidden state of the RNN
	# Compute the gradient of the mean squared error loss
	∇ = gradient(θ) do
		model(X[1]) # Warm-up the model
		sum(Flux.Losses.mse.([model(x)[1] for x ∈ X[2:end]], targets[2:end]))
	end
	Flux.update!(opt, θ, ∇) # Update the parameters
end

