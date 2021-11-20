function [main_chan_filtered] = LaPlacian(data,main_chan,chan_nearby, all_channels)
% Recives data matrix (N channels x T time steps), 
% the number of main channel and the vector of the nearby
% channels. It subtracts the nearby channels mean signal from the main one.
% the output is only the filtered signal from the main channel.
% If recives channels input as characters - need the 4th input- 
% vector of all channel names. TEST

main_chan_num=main_chan;
vec_num_nearby=chan_nearby;

    if ~isnumeric(main_chan)
        main_chan_num=find(all(ismember(all_channels,main_chan),2));
    end
    if ~isnumeric(chan_nearby)
            vec_num_nearby=find(all(ismember(all_channels,chan_nearby),2));
    end

    % Main calculation of the Laplacian
    main_chan_filtered = data(main_chan_num,:) - ...
                            mean(data(vec_num_nearby,:));
end