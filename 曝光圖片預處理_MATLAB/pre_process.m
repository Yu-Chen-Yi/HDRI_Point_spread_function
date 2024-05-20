clear; clc;

% 獲取目前資料夾內所有的 PNG 和 JPG 檔案
imageFiles = [dir('*.png'); dir('*.jpg')];

% 逐一處理每個檔案
for i = 1:length(imageFiles)
    % 讀取圖片
    filename = imageFiles(i).name;
    img = imread(filename);
    
    % 將 RGB 圖片轉換為灰階圖像
    grayImg = rgb2gray(img);
    
    % 對灰階圖像進行去噪處理（中值濾波，窗口大小為 20x20）
    denoisedImg = medfilt2(grayImg, [20, 20]);
    
    % 建立一個新的 3 通道圖片矩陣
    newImg = zeros(size(grayImg, 1), size(grayImg, 2), 3);
    
    % 將去噪後的灰階圖像複製到新矩陣的每個通道
    newImg(:, :, 1) = double(denoisedImg) / 255;  % 紅色通道
    newImg(:, :, 2) = double(denoisedImg) / 255;  % 綠色通道
    newImg(:, :, 3) = double(denoisedImg) / 255;  % 藍色通道
    
    % 建立新的檔名
    [~, name, ~] = fileparts(filename);
    result_filename = sprintf('result_%s.jpg', name);
    
    % 儲存去噪後的灰階圖像
    imwrite(newImg, result_filename);
end

disp('所有圖片已轉換並儲存完成。');