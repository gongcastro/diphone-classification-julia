include("src/Audios.jl")
include("src/Model.jl")

using .Audios, .Model
using Flux, Zygote
using ProgressMeter, Plots, Random, StatsBase
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

prop_train = 0.80
indices = 1:length(unique_diphones)
n_train = convert(Int32, floor(length(indices) * prop_train))
train_diphones = sample(unique_diphones, n_train, replace = false)
test_diphones = unique_diphones[findall(i -> i ∉ train_diphones, unique_diphones)]

diphones = getindex.(split.(wav_names, "_"), 2)
train_idx = shuffle(findall(i -> i ∈ train_diphones, diphones))
test_idx = shuffle(findall(i -> i ∈ test_diphones, diphones))

X = make_features(wav_files; pad = true, trans = StatsBase.ZScoreTransform)
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

idx = sample(1:length(X_train), length(X_train), replace = false)
X_train = X_train[idx]
y_train = y_train[idx]
data = Flux.DataLoader((X_train, y_train), batchsize = 4)

function loss(ŷ, y)
	Flux.reset!(model)
	return Flux.logitbinarycrossentropy(ŷ, y)
end

function accuracy(ŷ, y)
	return convert.(Int, ŷ .>= 0.5) .== y
end

begin
	model = Chain(LSTM(n_input => n_output), σ)
	epochs = 30
	opt = Adam(1e-3)
	opt_state = Flux.setup(opt, model)
	N = length(X_test)
	loss_hist = []
	acc_hist = []
end

for epoch ∈ 1:epochs
	losses = Float32[]
	local grads
	local acc_val
	local loss_val
	#local preds_vec
	local loss_vec
	local acc_vec
	preds_vec = []
	for (x, y) in zip(X_train, y_train)
		loss_vec = Float32[]
		acc_vec = []
		loss_val, grads = Flux.withgradient(model) do m
			preds = m(x)
			l = loss(preds, y)
			Zygote.ignore_derivatives() do
				Flux.reset!(model)
				acc_val = accuracy(preds, y)
			end
			return l
		end
		push!(loss_vec, loss_val)
		push!(acc_vec, acc_val)

	end
	mean_loss = mean(loss_vec)
	push!(loss_hist, mean_loss)

	mean_acc = mean(acc_vec[end])
	push!(acc_hist, mean_acc)

	Flux.update!(opt_state, model, grads[1])
	@info "Epoch $(epoch):" loss = mean_loss acc = mean_acc

end

mean_loss = mean(losses)
push!(loss_hist, mean_loss)

#mean_acc = mean(acc_val)
#push!(acc_hist, mean_acc)

#acc_vec = []
#@showprogress "Computing accuracy..." for x in X_test
#		acc_val = mean(convert.(Int, model(x) .>= 0.5) .== y_test)#
#	push!(acc_vec, acc_val)
#end
@info "Epoch $(epoch), loss = $(mean_loss)"


acc_val = mean([last(model(xi)) .>= 0.5 .== yi for (xi, yi) in zip(X_test, y_test)])

heatmap(X_train[1])

acc = make_predictions(model, X_test, y_test)

# plot training history
plot(mean.(acc_hist))
plot(loss_hist)

