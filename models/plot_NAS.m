clear all;
close all;
rng(1);

N = 5; %время
h = .1; %шаг дискретизации
M = 0:h:N; %сетка времени
u = zeros(length(M), 1); %управление
psi = zeros(length(M),1); %пси
x = zeros(length(M),2); %популяция жертв и хищников (1 столбец - жертвы, 2 столбец - хищники)
u(1) = 0;

%% коэффициенты и начальные значения:
% x(1,1) - начальное значение популяции жертв
% x(1,2) - начальное значение популяции хищников
% T1 - коэффициент Т1
% sko - СКО для генерации шума
% c - коэффициент затухания шума
% r - коэффициент ро в функции пси
% d - коэффициент d в функции пси
% alpha2 - коэффициент вымирания хищников
% beta1 - коэффициент вымирания жертв
% beta2 - коэффициент рождаемости хищников

% [x(1,1), x(1,2), T1, sko, c, r, d, alpha2, beta1, beta2] = deal(1, 1, .1, .01, .1, 1, 4, 0.12, 0.13, 0.05);

[x(1,1), x(1,2), T1, sko, c, r, d, alpha2, beta1, beta2] = deal(5, 10, .75, .03, .01, 5, 70, 0.1223, 0.1046, 0.0065);
x1_star = alpha2 / beta2;

% т.к. расчет начинается со 2 точки, то значения в первых двух точках совпадают
[x(2,1), x(2,2)] = deal(x(1,1), x(1,2));

% модель 3.29 в главе 3 монографии
psi(1) = x(1,1) - x1_star;
% % модель 3.30 в главе 3 монографии
% psi(1) = x(1,1) + r*x(1,2) - d;

%генерация массива точек шума с М=0 и заданным СКО
ksi = normrnd(0, sko, [1,size(M,2)]);

%цикл для расчета
for n=2:(length(M) - 1)
    % модель 3.29 в главе 3 монографии
    psi(n) = x(n,1) - x1_star;
    u(n) = (-psi(n)*(1 + T1) - c*(psi(n) + T1*psi(n-1))) / (h*x(n,1)) + beta1*x(n,2);
%     % модель 3.30 в главе 3 монографии
%     psi(n) = x(n,1) + r*x(n,2) - d;
%     u(n) = (-x(n,1) - c*(psi(n)+T1*psi(n-1)) - r*(x(n,2) + h*(-alpha2*x(n,2)+beta2*x(n,1)*x(n,2))) + d - T1*psi(n)) / (h*x(n,1)) + beta1*x(n,2);
    f1 = u(n)*x(n,1) - beta1*x(n,1)*x(n,2);
    f2 = -alpha2*x(n,2) + beta2*x(n,1)*x(n,2);

    x(n + 1,1) = x(n,1) + h*f1 + ksi(n+1) + c*ksi(n);
    x(n + 1,2) = x(n,2) + h*f2;
end


%% построение графика
xs = ones(length(M),4);
% расчет устоявшихся значений (2 случая)
% случай 1 (при beta2*d - alpha2 >= 0)
xs(:,1) = xs(:,1) .* (alpha2/beta2);
xs(:,2) = xs(:,2) .* ((beta2*d - alpha2)/ (r*beta2));
% случай 2 (при beta2*d - alpha2 < 0)
xs(:,3) = xs(:,3) .* d;
xs(:,4) = xs(:,4) .* 0;

% график функции управления u(t)
figure;
plot(M,u, 'Linewidth',2);
xlabel('Время, дни');
ylabel('Управление');
ax = gca;
ax.FontSize = 20;
grid on

% график популяций жертв, хищников и устоявшихся значений
figure;
hold on;
plot(M, x(:,1), 'g','Linewidth',3);
axis([0 N -inf inf]);
plot(M, x(:,2), 'r', 'Linewidth',3);
if beta2*d - alpha2 < 0
    plot(M, xs(:,3), 'k-.', 'Linewidth',2);
    plot(M, xs(:,4), 'k--', 'Linewidth',2);
else
    plot(M, xs(:,1), 'k-.', 'Linewidth',2);
    plot(M, xs(:,2), 'k--', 'Linewidth',2);
end
xlabel('Время, дни');
ylabel('Популяция, ед/л');
legend({'x_1', 'x_2', 'x_{1s}', 'x_{2s}'}, 'Location','best');
ax = gca;
ax.FontSize = 20;
grid on

% график функции psi
figure;
hold on;
plot(M,psi);
xlabel('t, дни');
ylabel('\psi(t)');
ax = gca;
ax.FontSize = 20;
grid on

% фазовый портрет
figure; 
hold on; 
plot(x(:,1), x(:,2),'Linewidth',3); 
plot(x(end,1),x(end,2), 'r.', 'MarkerSize', 20); 
xlabel('x_1'); 
ylabel('x_2');
xlim([0 35]);
ylim([9.8 10.3]);
legend({'             phase trajectory', 'finish'})
ax = gca;
ax.FontSize = 20;
grid on
