include("src/Audios.jl")
include("src/Model.jl")

using .Audios, .Model
using Flux, Plots, Random, StatsBase
using ProgressMeter
Plots.PythonPlotBackend()
Random.seed!(1234)

# Preprocess data ---------------------------------------------------------------

audio_path = "sounds/raw/"
wav_files = readdir(audio_path, join = true)[contains.(lowercase.(readdir(audio_path)), ".wav")]
wav_names = basename.(wav_files)

# unique diphones
unique_diphones = unique(getindex.(split.(wav_names, "_"), 2))
unique_speakers = unique(getindex.(split.(wav_names, "_"), 3))

# Train-Test split --------------------------------------------------------------

prop_train = 0.85
indices = 1:length(unique_diphones)
n_train = convert(Int32, floor(length(indices) * prop_train))
train_diphones = sample(unique_diphones, n_train, replace = false)
test_diphones = unique_diphones[findall(i -> i ∉ train_diphones, unique_diphones)]

diphones = getindex.(split.(wav_names, "_"), 2)
train_idx = findall(i -> i ∈ train_diphones, diphones)
test_idx = findall(i -> i ∈ test_diphones, diphones)

X = make_features(wav_files)
y, labels = make_targets(wav_files)

X_train = X[train_idx]
X_test = X[test_idx]

y_train, labels_train = y[train_idx], labels[train_idx]
y_test, labels_test = y[test_idx], labels[test_idx]

args = Args(epochs = 10)
n_input = size(X_train[1], 1)
n_hidden = trunc(Int, n_input / 2)
n_output = length(unique(y_train)) - 1

# model structure
model = Chain(
	LSTM(n_input => n_hidden),
	Dense(n_hidden => n_output),
	sigmoid,
)

opt = Adam(args.η)
opt_state = Flux.setup(opt, model)

loss_hist = Array{Float64, 1}(undef, args.epochs)
acc_hist = Array{Float64, 1}(undef, args.epochs)

function loss(input, label)
	Flux.logitbinarycrossentropy(input, label)
end

for epoch in 1:args.epochs
	Flux.reset!(model)

	if epoch == 1
		@info "Initialising model..."
	end

	N = length(X)
	loss_vec = Array{Float64, 1}(undef, N)
	acc_vec = Array{Float64, 1}(undef, N)

	# for each spectrogram-label pair
	@showprogress desc = "Computing gradient..." for (idx, d) in enumerate(zip(X_train, y_train))
		x, y = d
		# calculate gradient
		loss_val, ∇ = Flux.withgradient(model) do m
			result = m(x)
			loss(x, y)
		end

		# logging losses and accuracies
		acc_val = get_accuracy(model(x), y)
		loss_vec[idx] = loss_val
		acc_vec[idx] = acc_val
		# if NaN loss, skip iteration
		if !isfinite(loss_val)
			continue
		end

		# update the parameters
		Flux.update!(opt_state, model, ∇[1])
	end

	# logging mean losses and accuracies
	mean_loss = mean(filter(!isnan, loss_vec))
	mean_acc = mean(filter(!isnan, acc_vec))
	loss_hist[epoch] = mean_loss
	acc_hist[epoch] = mean_acc

	@info("Epoch $(epoch), accuracy = $(mean_acc), loss = $(mean_loss)")

	# stopping rule if accuracy gets very good
	if mean_acc > 0.95
		@warn("Stopping after $epoch epochs")
		break
	end
end
