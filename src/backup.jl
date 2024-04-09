include("Audios.jl")
include("Model.jl")

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

prop_train = 0.20
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

args = Args(epochs = 10, η = 1e-3)
n_input = size(X_train[1], 1)
n_hidden = trunc(Int, n_input / 2)
n_output = length(unique(y_train)) - 1

# model structure
model = Chain(
	RNN(n_input => n_hidden, tanh),
	Dense(n_hidden => n_output),
	sigmoid,
)

opt = Adam(args.η)
opt_state = Flux.setup(opt, model)

loss_hist = Array{Float64, 1}(undef, args.epochs)
acc_hist = Array{Float64, 1}(undef, args.epochs)

function loss(y_pred, y_true)
	Flux.reset!(model)
	return mean(Flux.logitbinarycrossentropy.(y_pred, y_true))
end


#################
# TRAINING LOOP #
#################

opt_state = Flux.setup(Adam(args.η), model)

for epoch in 1:args.epochs

	if epoch == 1
		@info "Initialising model..."
	end

	N = length(X_train)
	loss_vec = Array{Float64, 1}(undef, N)
	acc_vec = Array{Float64, 1}(undef, N)

	@showprogress for (idx, data) in enumerate(zip(X_train, y_train))
		input, label = data

		loss_val, grads = Flux.withgradient(model) do m
			# Any code inside here is differentiated.
			# Evaluation of the model and loss must be inside!
			result = m(input)
			loss(result, label)
		end
		loss_vec[idx] = loss_val
		acc_val = get_accuracy(model(input), label)
		acc_vec[idx] = acc_val

		# Detect loss of Inf or NaN. Print a warning, and then skip update!
		if !isfinite(loss_val)
			continue
		end

		Flux.update!(opt_state, model, grads[1])
	end

	mean_loss = mean(filter(!isnan, loss_vec))
	mean_acc = mean(filter(!isnan, acc_vec))
	loss_hist[epoch] = mean_loss
	acc_hist[epoch] = mean_acc

	@info("Epoch $(epoch), accuracy = $(mean_acc), loss = $(mean_loss)")

end
