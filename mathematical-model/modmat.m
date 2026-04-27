%% ===============================================================
% Modelo matemático desde LSTM entrenada - Linealización por tramos
% ===============================================================
clear; close all; clc;
%% --------- 0) Ajustes ----------
rng(0);                  %Semilla reproducible
Ts = 1;         % Tiempo de muestreo

%% --------- 1) Cargar e interpolar dataset ----------
load net                     
load C; load S      
load dataset_entreno.mat dataset

num_samples = size(dataset,1);  % Longitud de simulación (ajustable, antes 20001)
 
%% --------- 1) Cargar LSTM y dataset ----------
         
datasetn = normalize(dataset, "center", C, "scale", S);
X = datasetn(:,1:2);            % Entradas normalizadas
T_real = dataset(:,3:4);        % Salidas reales (desnormalizadas)
N = size(X,1);

% Extraer capa LSTM
lstmLayer = net.Layers(2);
numHidden = size(lstmLayer.InputWeights,1) / 4;

%% --------- Filtrado paso bajo para reducir aliasing ----------
fs = 1 / Ts;      % Frecuencia de muestreo (Hz)
fc = 0.1 * fs / 4;           % Frecuencia de corte (Hz), ajusta según tu sistema
order = 6;        % Orden del filtro Butterworth

Wn = fc / (fs/2); % Frecuencia normalizada para butter
[b, a] = butter(order, Wn, 'low');

% Filtrar cada columna de X sin desfase
X_filt = zeros(size(X));
for i = 1:size(X,2)
    X_filt(:,i) = filtfilt(b, a, X(:,i));
end

X = X_filt;
disp('Entrada filtrada con Butterworth paso bajo para reducir aliasing');

% Pesos y sesgos
Wf = lstmLayer.InputWeights(1:numHidden, :);
Wi = lstmLayer.InputWeights(numHidden+1:2*numHidden, :);
Wc = lstmLayer.InputWeights(2*numHidden+1:3*numHidden, :);
Wo = lstmLayer.InputWeights(3*numHidden+1:end, :);

Uf = lstmLayer.RecurrentWeights(1:numHidden, :);
Ui = lstmLayer.RecurrentWeights(numHidden+1:2*numHidden, :);
Uc = lstmLayer.RecurrentWeights(2*numHidden+1:3*numHidden, :);
Uo = lstmLayer.RecurrentWeights(3*numHidden+1:end, :);

bf = lstmLayer.Bias(1:numHidden);
bi = lstmLayer.Bias(numHidden+1:2*numHidden);
bc = lstmLayer.Bias(2*numHidden+1:3*numHidden);
bo = lstmLayer.Bias(3*numHidden+1:end);

%% --------- 2) Funciones de activación ----------
sigma_g = @(x) 1./(1+exp(-x));
sigma_c = @(x) tanh(x);
sigmoid_derivative = @(x) sigma_g(x).*(1 - sigma_g(x));
tanh_derivative = @(x) 1 - tanh(x).^2;
%% --------- 3) Linearización por tramos (con ponderación) ----------
x_lin = zeros(num_samples, numHidden*2);
A_sum = zeros(2*numHidden, 2*numHidden);
B_sum = zeros(2*numHidden, size(X,2));
weight_total = 0;  % Nuevo acumulador de pesos
h_pred = zeros(num_samples, numHidden);  % Guarda todos los estados h

% Estados iniciales
c0 = zeros(numHidden,1);
h0 = zeros(numHidden,1);

for t = 1:num_samples
    xi = X(mod(t-1,N)+1,:)';  % Entradas sin ruido (cíclicas)

    % --- Compuertas ---
    uf = Wf*xi + Uf*h0 + bf;
    ui = Wi*xi + Ui*h0 + bi;
    uc = Wc*xi + Uc*h0 + bc;
    uo = Wo*xi + Uo*h0 + bo;

    f = sigma_g(uf);
    i = sigma_g(ui);
    c_tilde = sigma_c(uc);
    o = sigma_g(uo);

    c_next = f .* c0 + i .* c_tilde;
    h_next = o .* sigma_c(c_next);

    % Derivadas vectoriales (sin diag())
    df = sigmoid_derivative(uf);   % [numHidden x 1]
    di = sigmoid_derivative(ui);
    dc_t = tanh_derivative(uc);
    do = sigmoid_derivative(uo);

    % Derivadas respecto a h (matrices) con broadcasting
    df_dh = Uf .* df;       % tamaño Uf es [numHidden x numHidden], df es [numHidden x 1]
    di_dh = Ui .* di;
    dc_tilde_dh = Uc .* dc_t;
    do_dh = Uo .* do;

    % Derivadas respecto a entrada u (matrices) con broadcasting
    df_du = Wf .* df;
    di_du = Wi .* di;
    dc_tilde_du = Wc .* dc_t;
    do_du = Wo .* do;

    % Jacobianos
    dc_dc = diag(f);
    dc_dh = diag(c_tilde)*di_dh + diag(i)*dc_tilde_dh + diag(c0)*df_dh;
    dc_du = diag(c_tilde)*di_du + diag(i)*dc_tilde_du + diag(c0)*df_du;

    dh_dc = diag(o .* (1 - tanh(c_next).^2)) * dc_dc;
    dh_dh = diag(sigma_c(c_next)) * do_dh + diag(o .* (1 - tanh(c_next).^2)) * dc_dh;
    dh_du = diag(sigma_c(c_next)) * do_du + diag(o .* (1 - tanh(c_next).^2)) * dc_du;

    % Matrices A y B
    A = [dc_dc, dc_dh; dh_dc, dh_dh];
    B = [dc_du; dh_du];

    % -------- Ponderación por norma del estado --------
    weight = norm([c_next; h_next]) + 1e-6;  % Evita división por cero
    A_sum = A_sum + weight * A;
    B_sum = B_sum + weight * B;
    weight_total = weight_total + weight;

    % Guardar estado y salida
    x_lin(t,:) = [c_next; h_next]';
    h_pred(t,:) = h_next';

    % Actualizar estado
    c0 = c_next;
    h0 = h_next;

    % Progreso (opcional)
    if mod(t, round(num_samples/10)) == 0
        fprintf('Progreso linearización: %d%%\n', round(100*t/num_samples));
    end
end

% Promedios ponderados
A_avg = A_sum / weight_total;
B_avg = B_sum / weight_total;

% Estado medio para punto de operación
x0_avg = mean(x_lin,1)';

%% --------- 4) Modelo en espacio de estados ---------
% Número de salidas (ny)
if exist('T_real','var')
    ny = size(T_real, 2);
else
    ny = 1; % Cambia este valor si tienes más salidas
end

% Número de estados
n_states = size(A_avg, 1);

% Número de entradas
nu = size(X, 2);

% Número de unidades ocultas
n_h = numHidden;

% Matrices C y D para espacio de estados
C_total = [zeros(ny, n_states - ny), eye(ny)];
D_total = zeros(ny, nu);

% Crear sistema discreto con Ts original
sys_ss = ss(A_avg, B_avg, C_total, D_total, Ts);

% Selector de h dentro del vector estado
Sel_h = [zeros(n_h, n_states - n_h), eye(n_h)];

G = tf(sys_ss);
%% --------- 5) Análisis del modelo linealizado ---------

fprintf('\n Evaluando punto de operación y linealización...\n');

% Revisa los valores propios del sistema linealizado
eig_vals = eig(A_avg);
%fprintf(' Autovalores de A_avg:\n');
%disp(eig_vals.');

if all(abs(eig_vals) < 1)
    disp('El sistema es ESTABLE (autovalores dentro del círculo unitario).');
else
    warning('El sistema NO es estable. Revisa la linealización.');
end

% Verificar controlabilidad y observabilidad del sistema completo
Co = ctrb(A_avg, B_avg);
Ob = obsv(A_avg, C_total);

rank_Co = rank(Co);
rank_Ob = rank(Ob);
n_states = size(A_avg,1);

fprintf('\n📋 Propiedades del sistema completo:\n');
fprintf('   Estados: %d\n', n_states);
fprintf('   Rango controlabilidad: %d de %d\n', rank_Co, n_states);
fprintf('   Rango observabilidad: %d de %d\n', rank_Ob, n_states);

if rank_Co < n_states
    warning('El sistema NO es completamente controlable.');
else
    disp('El sistema es completamente controlable.');
end

if rank_Ob < n_states
    warning('El sistema NO es completamente observable.');
else
    disp('El sistema es completamente observable.');
end

% Condición numérica
cond_Co = cond(Co);
cond_Ob = cond(Ob);

fprintf('📏 Condición numérica (controlabilidad): %.2e\n', cond_Co);
fprintf('📏 Condición numérica (observabilidad): %.2e\n', cond_Ob);

if cond_Co > 1e6
    warning('Matriz de controlabilidad mal condicionada (> 1e6).');
end
if cond_Ob > 1e6
    warning('Matriz de observabilidad mal condicionada (> 1e6).');
end

fprintf('\n➡️ Matrices listas para optimización con %d estados.\n', n_states);

%% --------- 5) BatchNorm recalibrado ---------
bnLayer = net.Layers(3);
gamma = bnLayer.Scale(:);     % 150 x 1
beta = bnLayer.Offset(:);     % 150 x 1
epsilon = 1e-5;

% Estadísticas de activación
mu_bn = mean(h_pred,1)';          % 150 x 1
sigma2_bn = var(h_pred,0,1)';     % 150 x 1

% Normalización + Affine transform
X_bn = (h_pred - mu_bn') ./ sqrt(sigma2_bn' + epsilon);  % 20001 x 150
h_bn = X_bn .* gamma' + beta';                            % 20001 x 150
%% --------- 6) Capa FullyConnected manual ---------
fcLayer = net.Layers(4);
W_fc = fcLayer.Weights;  % 2 x 150
b_fc = fcLayer.Bias;     % 2 x 1

y_fc = h_bn * W_fc.' + b_fc.';  % 20001 x 2

% Rehacer el selector Sel_h con dimensiones correctas
Sel_h = [zeros(n_h, n_states - n_h), eye(n_h)];

% Recalcular BatchNorm transform
BN_scale = diag(gamma ./ sqrt(sigma2_bn + epsilon));      
BN_bias  = beta - (gamma .* mu_bn ./ sqrt(sigma2_bn + epsilon)); 

W_total = W_fc * BN_scale;
b_total = W_fc * BN_bias + b_fc;

% Rehacer C_total y D_total
C_total = W_total * Sel_h;
D_total = zeros(ny, nu);
%% --------- 7) Desnormalizar salida final y calcular RMSE ---------
y_fc_desnorm = y_fc .* S(3:4) + C(3:4);   % 20001 x 2

rmse_no_noise = sqrt(mean((y_fc_desnorm - T_real).^2, 'all'));
fprintf('RMSE sin ruido: %.4f\n', rmse_no_noise);

range_y1 = max(T_real(:,1)) - min(T_real(:,1));
range_y2 = max(T_real(:,2)) - min(T_real(:,2));

rmse_rel_y1 = sqrt(mean((y_fc_desnorm(:,1) - T_real(:,1)).^2)) / range_y1 * 100;
rmse_rel_y2 = sqrt(mean((y_fc_desnorm(:,2) - T_real(:,2)).^2)) / range_y2 * 100;

fprintf('RMSE relativo salida 1: %.2f %%\n', rmse_rel_y1);
fprintf('RMSE relativo salida 2: %.2f %%\n', rmse_rel_y2);

%% --------- 8) Verificación de estabilidad ---------
figure;
pzmap(sys_ss);
title('Polos del sistema completo');
grid on;
%% --------- 8.1) Análisis de frecuencias naturales y amortiguamiento ---------
% Este bloque transforma los autovalores del sistema discreto a continuo
% para calcular frecuencia natural y razón de amortiguamiento, y detectar aliasing.

eig_z = eig(A_avg);                % Autovalores del sistema discreto (dominio z)
s_vals = log(eig_z) / Ts;         % Transformación al dominio continuo (s)

% Cálculo de frecuencia natural máxima total
s_vals_all = log(eig(A_avg)) / Ts;  % Ya lo tienes como 's_vals'
wn_all = abs(s_vals_all);           % Frecuencia natural (rad/s)
wn_max_total = max(wn_all);         % Frecuencia más alta del sistema
f_max_total = wn_max_total / (2*pi);  % En Hz

% Frecuencia de muestreo mínima recomendada (al menos 5 veces más alta)
fs_opt = 5 * f_max_total;
Ts_opt = 1 / fs_opt;

fprintf('\n Frecuencia natural más alta detectada: %.4f rad/s (%.4f Hz)\n', wn_max_total, f_max_total);
fprintf(' Frecuencia de muestreo mínima recomendada: %.4f Hz\n', fs_opt);
fprintf(' Tiempo de muestreo máximo recomendado (Ts): %.4f s\n', Ts_opt);

wn_vals = abs(s_vals);            % Frecuencia natural
zeta_vals = -real(s_vals) ./ wn_vals;  % Razón de amortiguamiento

% Límite real de Nyquist en rad/s
nyquist_limit = pi / Ts;

% Índices válidos (frecuencias naturales por debajo del límite de Nyquist)
valid_idx = wn_vals < nyquist_limit;

% Valores válidos (sin aliasing)
wn_valid = wn_vals(valid_idx);
zeta_valid = zeta_vals(valid_idx);

% Máxima frecuencia natural válida (rad/s)
wn_max = max(wn_valid);

% Convertir a Hz
f_max = wn_max / (2*pi);

% Frecuencia de muestreo recomendada para evitar aliasing (5x f_max)
fs_min = 5 * f_max;

% Tiempo de muestreo máximo recomendado
Ts_max = 1 / fs_min;

fprintf('\n Frecuencia natural máxima válida: %.4f rad/s (%.4f Hz)\n', wn_max, f_max);
fprintf(' Frecuencia de muestreo mínima recomendada para evitar aliasing: %.4f Hz\n', fs_min);
fprintf(' Tiempo de muestreo máximo recomendado (Ts): %.4f s\n', Ts_max);

% Contar cuántas frecuencias están por encima del límite de Nyquist
num_aliasing = sum(~valid_idx);
porcentaje_aliasing = 100 * num_aliasing / length(wn_vals);

fprintf(' Número de polos aliasados: %d de %d (%.2f%%)\n', ...
        num_aliasing, length(wn_vals), porcentaje_aliasing);

% Mostrar rangos válidos si hay valores válidos
if ~isempty(wn_valid)
    fprintf('\n Frecuencia natural válida: entre %.4f y %.4f rad/s\n', ...
            min(wn_valid), max(wn_valid));
    fprintf(' Razón de amortiguamiento válida: entre %.2f y %.2f\n', ...
            min(zeta_valid), max(zeta_valid));
else
    fprintf('\n No hay frecuencias naturales dentro del límite de Nyquist.\n');
end

% Verificar si hay aliasing
if any(~valid_idx)
    warning('Atención: Se detectaron frecuencias naturales mayores que la frecuencia de Nyquist (aliasing posible)');
else
    disp('Todas las frecuencias naturales están dentro del límite de Nyquist.');
end

%% --------- 8.2) Análisis de aliasing para varios tiempos de muestreo -------
Ts_values = linspace(0.1, 5, 50);  % Rango de tiempos de muestreo (ajusta si quieres)

aliasing_percent = zeros(size(Ts_values));
Ts_max_recommended = zeros(size(Ts_values));

% Los autovalores que tengo son para un Ts específico, usamos transformación log para analizar distintos Ts.

for k = 1:length(Ts_values)
    Ts_temp = Ts_values(k);
    s_vals_temp = log(eig_z) / Ts_temp;   % Transformar a dominio continuo con Ts temporal
    wn_vals_temp = abs(s_vals_temp);
    
    nyquist_limit_temp = pi / Ts_temp;
    
    valid_idx_temp = wn_vals_temp < nyquist_limit_temp;
    wn_valid_temp = wn_vals_temp(valid_idx_temp);
    
    if ~isempty(wn_valid_temp)
        wn_max_temp = max(wn_valid_temp);
        f_max_temp = wn_max_temp / (2*pi);
        fs_min_temp = 5 * f_max_temp;
        Ts_max_recommended(k) = 1 / fs_min_temp;
    else
        Ts_max_recommended(k) = NaN;
    end
    
    num_aliasing_temp = sum(~valid_idx_temp);
    aliasing_percent(k) = 100 * num_aliasing_temp / length(wn_vals_temp);
end

% Graficar resultados
figure;
yyaxis left
plot(Ts_values, aliasing_percent, '-o', 'LineWidth', 1.5);
ylabel('Porcentaje de polos aliasados [%]');
ylim([0 100]);

yyaxis right
plot(Ts_values, Ts_max_recommended, '-x', 'LineWidth', 1.5);
ylabel('Ts máximo recomendado [s]');
xlabel('Tiempo de muestreo Ts [s]');
title('Análisis de aliasing y tiempo de muestreo máximo recomendado');
grid on;
legend('Aliasado [%]', 'Ts máximo recomendado');


%% --------- 9) Comparaciones gráficas ---------

time = (0:num_samples-1)' * Ts;

figure;
subplot(2,1,1);
plot(time, T_real(:,1), 'k', 'LineWidth',1.2); hold on;
plot(time, y_fc_desnorm(:,1), 'b--', 'LineWidth',1);
grid on; xlabel('Tiempo [s]'); ylabel('Salida 1');
legend('Real','Linealizada');
title('Comparación de la salida 1');

subplot(2,1,2);
plot(time, T_real(:,2), 'k', 'LineWidth',1.2); hold on;
plot(time, y_fc_desnorm(:,2), 'b--', 'LineWidth',1);
grid on; xlabel('Tiempo [s]'); ylabel('Salida 2');
legend('Real','Linealizada');
title('Comparación de la salida 2');

% --------- Visualización de errores ---------
figure;
subplot(2,1,1);
plot(time, T_real(:,1) - y_fc_desnorm(:,1), 'b', 'LineWidth', 1.2);
xlabel('Tiempo [s]'); ylabel('Error y1');
grid on; title('Error de la salida 1');

subplot(2,1,2);
plot(time, T_real(:,2) - y_fc_desnorm(:,2), 'b', 'LineWidth', 1.2);
xlabel('Tiempo [s]'); ylabel('Error y2');
grid on; title('Error de la salida 2');

%% --------- 10) Modelo completo LSTM + BatchNorm + FullyConnected ---------

n_h = size(h_pred,2);            % Número de unidades ocultas (150)
n_states = size(x_lin,2);        % = 2*n_h
ny = size(W_fc,1);               % Número de salidas (2)
nu = size(X,2);                  % Número de entradas (2)

% Selector para h(k) dentro de x(k) = [c(k); h(k)]
n_states = size(A_avg, 1);    % 306
n_h = size(W_fc, 2);          % 150

% Asegurar que h(k) esté al final del vector de estado [ ... ; h(k) ]
Sel_h = [zeros(n_h, n_states - n_h), eye(n_h)];   % (150 x 306)

% Parámetros BatchNorm
BN_scale = diag(gamma ./ sqrt(sigma2_bn + epsilon));      
BN_bias  = beta - (gamma .* mu_bn ./ sqrt(sigma2_bn + epsilon)); 

% Total BatchNorm + FC
W_total = W_fc * BN_scale;                 % 2 x 150
b_total = W_fc * BN_bias + b_fc;           % 2 x 1

% Matrices para espacio de estados
C_total = W_total * Sel_h;                 % ny x n_states
D_total = zeros(ny, nu);                   % ny x nu

% Sistema completo
sys_ss_full = ss(A_avg, B_avg, C_total, D_total, Ts);
if sys_ss_full.Ts == 0
    disp('El sistema es CONTINUO.');
else
    fprintf('El sistema es DISCRETO con tiempo de muestreo Ts = %.4f segundos.\n', sys_ss_full.Ts);
end
disp('Sistema en espacio de estados COMPLETO (LSTM + BatchNorm + FC):');
disp(sys_ss_full);

eig_vals = eig(A_avg);         % Obtener los autovalores
is_stable = all(abs(eig_vals) < 1);  % Verificar que estén dentro del círculo unitario

if is_stable
    disp('El sistema 2 es ESTABLE.');
else
    disp('El sistema NO es estable.');
end

G_full = tf(sys_ss_full);
disp('Función de transferencia MIMO del modelo completo:');
%G_full

% Simulación del modelo completo
[y_full, ~, ~] = lsim(sys_ss_full, X, time);
y_full_desnorm = y_full .* S(3:4) + C(3:4);

rmse_full = sqrt(mean((y_full_desnorm - T_real).^2, 'all'));
fprintf('RMSE del modelo completo: %.4f\n', rmse_full);

% Gráficos
figure;
subplot(2,1,1);
plot(time, T_real(:,1), 'k', 'LineWidth',1.2); hold on;
plot(time, y_full_desnorm(:,1), 'm--', 'LineWidth',1.2);
grid on; xlabel('Tiempo [s]'); ylabel('Salida 1');
legend('Real','Modelo completo');
title('Comparación salida 1 - Modelo completo');

subplot(2,1,2);
plot(time, T_real(:,2), 'k', 'LineWidth',1.2); hold on;
plot(time, y_full_desnorm(:,2), 'm--', 'LineWidth',1.2);
grid on; xlabel('Tiempo [s]'); ylabel('Salida 2');
legend('Real','Modelo completo');
title('Comparación salida 2 - Modelo completo');
% Guardar las matrices linealizadas para usar en la optimización
save('linear_model.mat', 'A_avg', 'B_avg', 'C_total', 'D_total', 'Ts');

%% --------- 11) Verificación, reducción y optimización extendida ---------

fprintf('\n Verificación estructural y reducción del sistema\n');

% Crear sistema original
sys_ss = ss(A_avg, B_avg, C_total, D_total, Ts);

% Obtener valores singulares de energía para visualización
[~, g] = balreal(sys_ss);

% Mostrar el perfil de energía
figure;
semilogy(g, 'o-');
xlabel('Número de estado');
ylabel('Valor singular de energía');
title('Perfil de valores singulares (gramianos balanceados)');
grid on;

% Verificar controlabilidad y observabilidad del sistema original
rank_ctrb = rank(ctrb(A_avg, B_avg));
rank_obsv = rank(obsv(A_avg, C_total));

fprintf('Rango de controlabilidad (original): %d de %d\n', rank_ctrb, size(A_avg,1));
fprintf('Rango de observabilidad (original): %d de %d\n', rank_obsv, size(A_avg,1));

% Elegir orden automáticamente según umbral o definirlo manualmente
umbral = 1e-15;
orden_deseado = sum(g >= umbral);  % número de estados con energía significativa

fprintf('Estados con baja energía detectados: %d\n', sum(g < umbral));

% Limitar el orden al máximo permitido (por ejemplo, 24)
orden_maximo = 8;
orden_deseado = min(orden_deseado, orden_maximo);

fprintf('Reduciendo sistema a orden %d (máximo permitido: %d)\n', ...
    orden_deseado, orden_maximo);

% Reducir el sistema automáticamente
sys_red = balred(sys_ss, orden_deseado);

fprintf('Sistema reducido de %d a %d estados tras balred.\n', ...
    size(sys_ss.A,1), size(sys_red.A,1));

% Verificar controlabilidad del sistema reducido
Co_red = ctrb(sys_red.A, sys_red.B);
rank_red = rank(Co_red);
fprintf('Rango de controlabilidad (reducido): %d de %d\n', ...
    rank_red, size(sys_red.A,1));

% Verificar condición numérica de controlabilidad y observabilidad
cond_Co_red = cond(Co_red);
cond_Ob_red = cond(obsv(sys_red.A, sys_red.C));
fprintf('Condición numérica (controlabilidad reducida): %e\n', cond_Co_red);
fprintf('Condición numérica (observabilidad reducida): %e\n', cond_Ob_red);

% (Opcional) Verificar estabilidad del sistema reducido
eigs_red = eig(sys_red.A);
if all(abs(eigs_red) < 1)  % Para sistema discreto
    fprintf('El sistema reducido es estable.\n');
else
    fprintf('El sistema reducido NO es estable.\n');
end

%% --------- 12) Optimización extendida del sistema limpio (versión reducida) ---------

% Extraer matrices del sistema reducido
A_fixed = sys_red.A;
B_fixed = sys_red.B;
C_total = sys_red.C;
%D_total = sys_red.D;
D_total = zeros(size(sys_red.D));  % o simplemente [ny, nu] si los conoces

% Dimensiones
n_states = size(A_fixed, 1);
nu = size(B_fixed, 2);
ny = size(C_total, 1);

% Índices de A a optimizar: TODOS los elementos
A_idx = 1:numel(A_fixed);  % todos los elementos de A

% Índices de B que vamos a optimizar: elegir solo primeras columnas
%cols_to_opt = 1:min(3, nu);  % optimiza hasta 3 columnas de B (o menos si nu < 3)
cols_to_opt = 1:nu;  % optimiza todas las columnas

B_idx = [];
for col = cols_to_opt
    B_idx = [B_idx, sub2ind([n_states, nu], (1:n_states), col * ones(1, n_states))];
end

% Construir vector de parámetros iniciales
initial_params = double([
    A_fixed(A_idx(:));
    B_fixed(B_idx(:))
]);

% Verificar dimensiones de entrada (en caso de que falten columnas)
if size(X, 2) < nu
    X = [X, zeros(size(X, 1), nu - size(X, 2))];
end

% Ventana gráfica para elegir el optimizador
[opt_choice_idx, ok] = listdlg( ...
    'PromptString', 'Seleccione el optimizador:', ...
    'SelectionMode', 'single', ...
    'ListString', {'fminunc (quasi-newton)', 'fmincon (SQP)'}, ...
    'Name', 'Selector de optimizador');

if ~ok
    error('Selección cancelada por el usuario.');
end

opt_choice = opt_choice_idx;  % 1 o 2 según la selección

% Definir función objetivo (igual para ambos)
objective_fun = @(params) objective_reduced(params, A_fixed, B_fixed, ...
    A_idx, B_idx, ...
    C_total, D_total, Ts, X, T_real, S, C, time);

switch opt_choice
    case 1
        % Opciones para fminunc
        options = optimoptions('fminunc', ...
            'Algorithm', 'quasi-newton', ...
            'Display', 'iter', ...
            'MaxIterations', 500);
        
        % Ejecutar optimización con fminunc
        [opt_params, final_rmse, exitflag, output] = fminunc(objective_fun, initial_params, options);
        fprintf('Optimización con fminunc finalizada.\n');
        
    case 2
        % Opciones para fmincon
        options = optimoptions('fmincon', ...
            'Algorithm', 'sqp', ...
            'Display', 'iter', ...
            'MaxIterations', 500);
        
        % Límites para parámetros (ejemplo: sin límites)
        lb = [];
        ub = [];
        % Sin restricciones no lineales
        nonlcon = [];
        
        % Ejecutar optimización con fmincon
        [opt_params, final_rmse, exitflag, output] = fmincon( ...
            objective_fun, initial_params, [], [], [], [], lb, ub, nonlcon, options);
        fprintf('Optimización con fmincon finalizada.\n');
        
    otherwise
        error('Opción inválida. Por favor, ingrese 1 o 2.');
end

% Reconstruir A_opt y B_opt
A_opt = A_fixed;
A_opt(A_idx) = opt_params(1 : numel(A_idx));

% Índice desde donde empieza B en el vector opt_params
startB = numel(A_idx) + 1;
endB = startB + numel(B_idx) - 1;

B_opt = B_fixed;
B_opt(B_idx) = opt_params(startB : endB);

%% --------- 13) Verificación final, simulación y graficado ---------
% Simulación
sys_opt = ss(A_opt, B_opt, C_total, D_total, Ts);
[y_opt, ~, ~] = lsim(sys_opt, X, time);


eig_vals = eig(A_opt);         % Obtener los autovalores
is_stable = all(abs(eig_vals) < 1);  % Verificar que estén dentro del círculo unitario

if is_stable
    disp('El sistema ultimo es ESTABLE.');
else
    disp('El sistema NO es estable.');
end

% Desnormalizar
y_opt_desnorm = y_opt .* S(3:4) + C(3:4);

fprintf('RMSE después de optimización: %.4f\n', final_rmse);

% Graficar
figure;
subplot(2,1,1);
plot(time, T_real(:,1), 'k', time, y_opt_desnorm(:,1), 'r--', 'LineWidth', 1.2);
xlabel('Tiempo [s]'); ylabel('Salida 1');
legend('Real','Optimizado');
grid on;
title('Comparación salida 1-Unnormalized');

subplot(2,1,2);
plot(time, T_real(:,2), 'k', time, y_opt_desnorm(:,2), 'r--', 'LineWidth', 1.2);
xlabel('Tiempo [s]'); ylabel('Salida 2');
legend('Real','Optimizado');
grid on;
title('Comparación salida 2-Unnormalized');

% Guardar
save('linear_model_optimized_extended.mat', ...
     'A_opt', 'B_opt', 'C_total', 'D_total', 'Ts');
save('datos_mpc.mat', 'X', 'T_real', 'S', 'C', ...
     'W_fc', 'b_fc', 'gamma', 'beta', 'mu_bn', 'sigma2_bn', 'epsilon', ...
     'h_pred', 'x_lin', 'Ts', 'num_samples');

disp('Proceso completado.');
%% --------- 14) Verificación final y validación del sistema optimizado ---------

fprintf('\n==============================\n');
fprintf(' Validación del sistema optimizado\n');
fprintf('==============================\n');

% 1. Estabilidad del sistema
eig_vals = eig(A_opt);
if all(abs(eig_vals) < 1)
    fprintf(' El sistema optimizado es ESTABLE (|λ| < 1).\n');
else
    fprintf(' El sistema optimizado NO es estable (|λ| ≥ 1).\n');
end

% 2. Controlabilidad y observabilidad
Co = ctrb(A_opt, B_opt);
Ob = obsv(A_opt, C_total);  % Usa C_opt si también lo optimizaste

rank_Co = rank(Co);
rank_Ob = rank(Ob);
n_states = size(A_opt, 1);

fprintf('Rango de controlabilidad: %d de %d\n', rank_Co, n_states);
fprintf('Rango de observabilidad: %d de %d\n', rank_Ob, n_states);

% 3. Condición numérica
cond_Co = cond(Co);
cond_Ob = cond(Ob);
fprintf(' Condición de controlabilidad: %.2e\n', cond_Co);
fprintf(' Condición de observabilidad: %.2e\n', cond_Ob);

G = tf(sys_opt)
% 5. Cálculo del error RMSE
rmse = sqrt(mean((y_opt_desnorm - T_real).^2, 'all'));
fprintf('RMSE final del sistema optimizado: %.4f\n', rmse);

% MAE
mae = mean(abs(y_opt_desnorm - T_real), 'all');
fprintf('MAE final: %.4f\n', mae);

% R^2 (por salida)
R2 = zeros(ny,1);
for i = 1:ny
    SS_res = sum((T_real(:,i) - y_opt_desnorm(:,i)).^2);
    SS_tot = sum((T_real(:,i) - mean(T_real(:,i))).^2);
    R2(i) = 1 - SS_res/SS_tot;
end
fprintf('R^2 final por salida: y1=%.4f, y2=%.4f\n', R2(1), R2(2));

% Error relativo global (%)
range_y = max(T_real) - min(T_real);  % rango por cada salida
error_rel = rmse ./ range_y * 100;
fprintf('Error relativo global por salida: y1=%.2f%%, y2=%.2f%%\n', error_rel(1), error_rel(2));

% 6. Gráficos comparativos
%% =====================================================
%  FIGURA SUBPLOTS – h1 y h4 (FORMATO Q1)
% =====================================================

%% Parámetros visuales globales
lw_plot  = 2.8;     % Grosor líneas
lw_axes  = 2.0;     % Grosor ejes
fs_axes  = 18;      % Tamaño números ejes
fs_label = 26;      % Etiquetas ejes
fs_title = 30;      % Título

%% Crear figura vertical grande
f = figure('Units','inches', ...
           'Position',[0 0 10 12], ...
           'PaperUnits','inches', ...
           'PaperPosition',[0 0 10 12], ...
           'PaperSize',[10 12], ...
           'Color','w', ...
           'ToolBar','none', ...
           'MenuBar','none');

%% =====================================================
%  SUBPLOT 1 — h1
% =====================================================

ax1 = subplot(2,1,1);
hold(ax1,'on')

plot(time, T_real(:,1), 'k', 'LineWidth', lw_plot)
plot(time, y_opt_desnorm(:,1), 'r--', 'LineWidth', lw_plot)

xlabel('Time [s]','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold')

ylabel('$h_1$','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold')

title('Comparison of Output 1','Interpreter','latex', ...
      'FontSize',fs_title,'FontWeight','bold')

legend('Actual','Optimized', ...
       'Interpreter','latex', ...
       'FontSize',fs_axes, ...
       'Location','best')

set(ax1, ...
    'FontSize',fs_axes, ...
    'LineWidth',lw_axes, ...
    'Box','on', ...
    'TickLabelInterpreter','latex')

grid on

%% =====================================================
%  SUBPLOT 2 — h4
% =====================================================

ax2 = subplot(2,1,2);
hold(ax2,'on')

plot(time, T_real(:,2), 'k', 'LineWidth', lw_plot)
plot(time, y_opt_desnorm(:,2), 'r--', 'LineWidth', lw_plot)

xlabel('Time [s]','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold')

ylabel('$h_4$','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold')

title('Comparison of Output 2','Interpreter','latex', ...
      'FontSize',fs_title,'FontWeight','bold')

legend('Actual','Optimized', ...
       'Interpreter','latex', ...
       'FontSize',fs_axes, ...
       'Location','best')

set(ax2, ...
    'FontSize',fs_axes, ...
    'LineWidth',lw_axes, ...
    'Box','on', ...
    'TickLabelInterpreter','latex')

grid on

%% Ajuste fino márgenes tipo journal
drawnow
ti1 = ax1.TightInset;
ax1.Position = [ti1(1), 0.57, 1 - ti1(1) - ti1(3), 0.32];

ti2 = ax2.TightInset;
ax2.Position = [ti2(1), 0.10, 1 - ti2(1) - ti2(3), 0.32];

%% Exportar en vectorial para LaTeX
exportgraphics(f,'ComparisonOutputs.pdf','ContentType','vector')



%% =====================================================
%  FIGURA ÚNICA – ERRORES h1 y h4 (FORMATO Q1 FINAL)
% =====================================================

%% Parámetros visuales globales
lw_plot  = 2.8;
lw_axes  = 2.0;
fs_axes  = 18;
fs_label = 26;
fs_title = 30;

%% Calcular errores
error1 = T_real(:,1) - y_opt_desnorm(:,1);
error2 = T_real(:,2) - y_opt_desnorm(:,2);

%% Escala simétrica común
max_err = max(abs([error1; error2]));
ylim_val = 1.05 * max_err;

%% Crear figura vertical
f = figure('Units','inches', ...
           'Position',[0 0 10 12], ...
           'PaperUnits','inches', ...
           'PaperPosition',[0 0 10 12], ...
           'PaperSize',[10 12], ...
           'Color','w', ...
           'ToolBar','none', ...
           'MenuBar','none');

%% =========================
%  Subplot 1 – Error h1
% =========================
ax1 = subplot(2,1,1);
plot(time, error1, 'b', 'LineWidth', lw_plot);
hold on
yline(0,'k--','LineWidth',1.5);

xlabel('Time [s]','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')
ylabel('Error $h_1$','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')
title('Prediction Error -- $h_1$','Interpreter','latex','FontSize',fs_title,'FontWeight','bold')

ylim([-ylim_val ylim_val])
grid on

set(ax1,'FontSize',fs_axes,'LineWidth',lw_axes,...
    'Box','on','TickLabelInterpreter','latex')

%% =========================
%  Subplot 2 – Error h4
% =========================
ax2 = subplot(2,1,2);
plot(time, error2, 'b', 'LineWidth', lw_plot);
hold on
yline(0,'k--','LineWidth',1.5);

xlabel('Time [s]','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')
ylabel('Error $h_4$','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')
title('Prediction Error -- $h_4$','Interpreter','latex','FontSize',fs_title,'FontWeight','bold')

ylim([-ylim_val ylim_val])
grid on

set(ax2,'FontSize',fs_axes,'LineWidth',lw_axes,...
    'Box','on','TickLabelInterpreter','latex')

%% Ajuste fino tipo journal
drawnow
ti1 = ax1.TightInset;
ax1.Position = [ti1(1), 0.57, 1 - ti1(1) - ti1(3), 0.32];

ti2 = ax2.TightInset;
ax2.Position = [ti2(1), 0.10, 1 - ti2(1) - ti2(3), 0.32];

%% Exportar en vectorial
exportgraphics(f,'PredictionErrors.pdf','ContentType','vector')



% 7. Guardar modelo optimizado
save('modelo_optimizado_final.mat', 'A_opt', 'B_opt', 'C_total', 'D_total', 'Ts');

fprintf('Modelo optimizado guardado como modelo_optimizado_final.mat\n');
%% --------- Comprobación de aliasing (frecuencias rápidas) ---------

fprintf('\n Comprobación de aliasing en el modelo optimizado:\n');

% Calcular polos del sistema optimizado
poles = eig(A_opt);

% Convertir polos a frecuencia (rad/sample)
angles_rad = abs(angle(poles));  % Argumento (fase)
frequencies_hz = angles_rad / (2*pi*Ts);  % En Hz

% Frecuencia de Nyquist
Fs = 1/Ts;
f_nyquist = Fs / 2;

% Revisar si hay frecuencias cercanas al límite de Nyquist
frecuencia_umbral = 0.8 * f_nyquist;  % Umbral de alerta (80%)
frecuencia_maxima = max(frequencies_hz);

fprintf('Frecuencia de muestreo: %.2f Hz\n', Fs);
fprintf('Frecuencia de Nyquist: %.2f Hz\n', f_nyquist);
fprintf('Máxima frecuencia de polo detectada: %.2f Hz\n', frecuencia_maxima);

if frecuencia_maxima >= frecuencia_umbral
    warning(['Posible aliasing: hay dinámicas cercanas al límite de Nyquist.\n', ...
             '   Considera reducir Ts (aumentar Fs) o filtrar el sistema.']);
else
    fprintf('No se detectan problemas evidentes de aliasing.\n');
end

%% --------- Bode MIMO del sistema optimizado ---------

sys_bode = sys_opt;
opts = bodeoptions;
opts.Grid = 'on';
opts.Title.FontSize = 12;

[ny, nu] = size(sys_bode);

% --- Figura para y1 ---
figure('Units','inches','Position',[1 1 8 6])
t1 = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

for j = 1:nu
    ax = nexttile;
    bode(sys_bode(1,j), opts)
    title(sprintf('y_1 / u_%d', j))
end

sgtitle('Bode diagrams of y_1','FontSize',14)
exportgraphics(gcf,'Bode_y1.png','Resolution',300);

% --- Figura para y2 ---
figure('Units','inches','Position',[1 1 8 6])
t2 = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

for j = 1:nu
    ax = nexttile;
    bode(sys_bode(2,j), opts)
    title(sprintf('y_2 / u_%d', j))
end

sgtitle('Bode diagrams of y_2','FontSize',14)
exportgraphics(gcf,'Bode_y2.png','Resolution',300);
