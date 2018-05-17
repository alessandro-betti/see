clear variables;
close all;

% ----------------
% CONFIGURATION
% ----------------
figdest_compare_params = '/Users/mela/Desktop/NIPS_PLOTS/data_compare_params/';
color_pattern_comp = 1;
expdirs_compare_params = {
    '/Users/mela/Desktop/NIPS_PLOTS/data_compare_params/notstab_notreal', ...
    '/Users/mela/Desktop/NIPS_PLOTS/data_compare_params/notstab_real', ...
    '/Users/mela/Desktop/NIPS_PLOTS/data_compare_params/stab_notreal', ...
    '/Users/mela/Desktop/NIPS_PLOTS/data_compare_params/stab_real'
    };
figdest5x5 = '/Users/mela/Desktop/NIPS_PLOTS/data_5x5/';
color_patternf = 2;
expdirs5x5 = {
    '/Users/mela/Desktop/NIPS_PLOTS/data_5x5/car', ...
    '/Users/mela/Desktop/NIPS_PLOTS/data_5x5/matrix', ...
    '/Users/mela/Desktop/NIPS_PLOTS/data_5x5/skater'
    };
figdest11x11 = '/Users/mela/Desktop/NIPS_PLOTS/data_11x11/';
expdirs11x11 = {
    '/Users/mela/Desktop/NIPS_PLOTS/data_11x11/car', ...
    '/Users/mela/Desktop/NIPS_PLOTS/data_11x11/matrix', ...
    '/Users/mela/Desktop/NIPS_PLOTS/data_11x11/skater'
    };
figdestblur = '/Users/mela/Desktop/NIPS_PLOTS/data_blur/';
color_patternb = 3;
expdirsblur = {
    '/Users/mela/Desktop/NIPS_PLOTS/data_blur/slow_stab_real', ...
    '/Users/mela/Desktop/NIPS_PLOTS/data_blur/fast_stab_real', ...    
    '/Users/mela/Desktop/NIPS_PLOTS/data_blur/none_stab_real'
    };

explabelscompare = {
    'no-stability, no-reality', ...
    'no-stability, reality', ...
    'stability, no-reality', ...
    'stability, reality'
    };

explabelsb = {
    'slow', ...
    'fast', ...
    'faster (none)'
    };

explabelsv = {
    'car', ...
    'matrix', ...
    'skater'
    };

no_y_label = true;

figdest = [figdest_compare_params]; expdirs = expdirs_compare_params; explabels = explabelscompare; color_pattern = color_pattern_comp;
%figdest = [figdest5x5]; expdirs = expdirs5x5; explabels = explabelsv;  color_pattern = color_patternf;
%figdest = [figdest11x11]; expdirs = expdirs11x11; explabels = explabelsv;  color_pattern = color_patternf;
%figdest = [figdestblur]; expdirs = expdirsblur; explabels = explabelsb; color_pattern = color_patternb;

density_window = 1000; % if <= 1 then 'moving average', if < 0 then 'gaussian smoothing'
measure_labels = { 
    'MI', ...
    'CAL: C.Entropy', ...
    '||q||^2', ...
    'CAL: MI', ...
    'Resets', ...
    'CAL: Action', ...
    '1-Blurring Factor'
    };
measure_file_labels = {
    'mi', ...
    'ca-ce', ...
    'normq', ...
    'ca-mi', ...    
    'resets', ...
    'ca-action', ...
    'rho'
    };
measure_cols = [6,15,8,18,25,14,26] + 1;
if no_y_label == true
    for i = 1:length(measure_file_labels)
        measure_file_labels{i} = [measure_file_labels{i} '_noylabel'];
    end
end
% ----------------

% reading data
ne = length(expdirs);
data = cell(ne,1);
for i = 1:ne
    for j = 1:5
        if j == 1
            csv = csvread([expdirs{i} '/' num2str(j) '/log_layer0.txt'],1,0);
        else
            csv2 = csvread([expdirs{i} '/' num2str(j) '/log_layer0.txt'],1,0);
            csv = csv + csv2;
        end
    end
    data{i} = csv / 5.0;
end

% getting data
x = cell(ne,1);
y = cell(ne,length(measure_cols));
for i = 1:ne
    x{i} = data{i}(:,1); % steps
    for j = 1:length(measure_cols)
        y{i,j} = data{i}(:,measure_cols(j)); % data
        if strcmp(measure_labels{j}, 'Resets')
            y{i,j} = to_density(y{i,j},density_window);
        end
    end
end

% plotting
for j = 1:length(measure_cols)
    [xm,ym] = prepare_data(x,y,j);
    if strcmp(measure_labels{j}, 'Resets')
        xm = xm * density_window;
    end    
    plotfig(xm,ym,explabels,measure_labels{j},figdest,measure_file_labels{j}, color_pattern, no_y_label);
end

function plotfig(x, y, exp_labels, y_label, fig_dest, filename, color_pattern, no_y_label)
    styles = {'-', '--', ':', '-.'};
    widths = {3, 3, 3, 3};
    if color_pattern == 1
        colors = {[0.749019622802734 0 0.749019622802734], [0.87058824300766 0.490196079015732 0], [0 0 1], [0.501960813999176 0.501960813999176 0.501960813999176]};
    else
        if color_pattern == 2
            colors = {[0 0.498039215803146 0], [1 0 0], [0 0 1]};
        else
            if color_pattern == 3
                colors = {[0 0.447058826684952 0.74117648601532], [0.749019622802734 0 0.749019622802734], [1 0 0]};
            else
                colors = {[0.929411768913269 0.694117665290833 0.125490203499794], [0.635294139385223 0.0784313753247261 0.184313729405403]};
            end
        end
    end
    f = figure;
    ax = axes(f,'FontSize',22,'Box','on');
    hold all;
    ne = size(y,1);
    
    for i = 1:ne
        plot(x,y(i,:), 'DisplayName', exp_labels{i}, 'LineWidth', widths{mod(i-1,length(widths))+1}, 'LineStyle', styles{mod(i-1,length(styles))+1}, 'Color', colors{mod(i-1,length(colors))+1});
    end
    xlim([min(x),max(x)]);
    ylim([min(y(:))*0.95, max(y(:))*1.05]);
    xlabel('Frame'); %,'Interpreter','latex');
    if no_y_label == false
        ylabel(y_label); %,'Interpreter','latex');
    end
    legend('show', 'Location', 'best');
    
    outerpos = ax.OuterPosition;
    ti = ax.TightInset; 
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    fixFactor = 0.98;
    ax.Position = [left bottom ax_width*fixFactor ax_height*fixFactor];
    
    f.PaperPositionMode = 'auto';
    fig_pos = f.PaperPosition;
    f.PaperSize = [fig_pos(3) fig_pos(4)];

    saveas(f, [fig_dest '/' filename '.fig']);
    saveas(f, [fig_dest '/' filename '.pdf']);
end

function [xm,ym] = prepare_data(x, y, measure)
    T = length(y{1,measure});
    ne = size(y,1);
    for i = 1:ne
        T = min(T,length(y{i,measure}));
    end
    xm = zeros(ne,T);
    ym = zeros(ne,T);
    for i = 1:ne
        ym(i,:) = y{i,measure}(1:T);
        if i == 1
            xm = x{i}(1:T);
        end
    end
end

function yp = to_density(y,w)
    yp = zeros(max(length(y),2),1);
    ma = yp(1);
    if w < 0.0
        %gaussianf = gausswin(-w);    
        sigma = w; 
        a = (1:floor(w/2)); 
        b = exp(-(a.^2)/(2*sigma*sigma));
        gaussianf = [b,1.0,b];
        gaussianf = gaussianf ./ sum(gaussianf);
        yp = filter(gaussianf,1,y);    
        return
    end
    k = 1;
    for i = 1:w:length(y)
        if w <= 1
            ma = (1.0-w)*ma + w*y(i);
            yp(i) = ma;
        else
            if i + w - 1 <= length(y)
                s = double(sum(y(i:i+w-1)));
                d = s / w;
            else
                if i == 1
                    w = length(y);
                    s = double(sum(y(i:i+w-1)));
                    d = s / w; 
                    yp(1) = d;
                    yp(2) = d;
                    yp = yp(1:2);
                else
                    if i == 2
                        yp(2) = yp(1);
                        yp = yp(1:2);
                    else
                        yp = yp(1:i-1);
                    end
                end
                break
            end 
            
            yp(k) = d;
            k = k + 1;
        end
    end
    
    yp = yp(1:k-1);
end


