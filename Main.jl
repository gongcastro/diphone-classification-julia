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

prop_train = 0.10
indices = 1:length(unique_diphones)
n_train = convert(Int32, floor(length(indices) * prop_train))
train_diphones = sample(unique_diphones, n_train, replace = false)
test_diphones = unique_diphones[findall(i -> i ∉ train_diphones, unique_diphones)]

diphones = getindex.(split.(wav_names, "_"), 2)
train_idx = shuffle(findall(i -> i ∈ train_diphones, diphones))
test_idx = shuffle(findall(i -> i ∈ test_diphones, diphones))

X = make_features(wav_files; pad = true)
y, labels = make_targets(wav_files)

X_train = X[train_idx]
X_test = X[test_idx]

y_train, labels_train = y[train_idx], labels[train_idx]
y_test, labels_test = y[test_idx], labels[test_idx]

args = Args(epochs = 100, η = 1e-3)
n_input = size(X_train[1], 1)
n_hidden = trunc(Int, n_input / 2)
n_output = length(unique(y_train)) - 1


#################
# TRAINING LOOP #
#################

begin
	model = Chain(
		LSTM(n_input => n_hidden),
		Dense(n_hidden => n_output, sigmoid),
		sigmoid,
	)
	epochs = 10
	opt_state = Flux.setup(ADAM(0.001), model)
	N = length(X_test)
	loss_hist = []
	acc_hist = []
	ps_hist = []
end

for epoch ∈ 1:epochs

	loss_vec = []
	acc_vec = []
	local ∇
	@showprogress for (x, y) in zip(X_train, y_train)

		loss_val, ∇ = Flux.withgradient(model) do m
			Flux.reset!(model)
			loss_val = Flux.Losses.logitbinarycrossentropy(last(m(x)), y)
			return loss_val
		end

		if !isfinite(loss_val)
			continue
		end
		push!(loss_vec, loss_val)
	end
	mean_loss = mean(loss_val)

	Flux.update!(opt_state, model, ∇[1])


	acc_vec = []
	for (x, y) in zip(X_test, y_test)
		acc_val = mean(get_accuracy(model(x), y))
		push!(acc_vec, acc_val)
	end
	mean_acc = mean(filter(!isnan, acc_vec))


	push!(loss_hist, mean_loss)
	push!(acc_hist, mean_acc)

	@info "Epoch $(epoch), loss = $(mean_loss), acc = $(mean(acc_vec))"
end

acc_val = mean([last(model(xi)) .>= 0.5 .== yi for (xi, yi) in zip(X_test, y_test)])

heatmap(X_train[1])

acc = make_predictions(model, X_test, y_test)

# plot training history
plot(mean.(acc_hist))
plot(loss_hist)

