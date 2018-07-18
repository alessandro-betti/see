n1 = 10; % repetitions

x1 = 0.55; % average grade in class A
s1 = 0.13; % std dev of exam grade in class A

x2 = 0.62; % average grade in class B
s2 = 0.02; % std dev of exam grade in class B

SE = sqrt(s1^2/n1 + s2^2/n1);
DF = (n1 - 1) + (n1 - 1);
tscore = abs(((x1 - x2)-0)/SE)
ci = 0.95;
alpha = 1 - ci;
t95 = tinv(1-alpha/2, DF)

if tscore > t95, disp('Significantly different!'); else disp('Same stuff...'); end

f = tcdf(tscore,DF)-tcdf(-tscore,DF)
