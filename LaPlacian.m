function [data_bef,data_aft] = LaPlacian(data,main_chan,chan_nearby, all_channels)
% Recives data matrix (N channels x T time steps), 
% the number of main channel/s and the vector 
% (or matrix in case of a few filters, in which each row is composed) of the nearby
% channels. It subtracts the nearby channels mean signal from the main one.
% the output is the data matrix without and after the change in the
% filtered signal from the main channel.
% If recives channels input as characters - need the 4th input- 
% vector of all channel names.
data_bef=data; data_aft=data_bef;
main_chan_num=main_chan;
num_nearby=chan_nearby;
len_data_time=size(data,2);

    if ~isnumeric(main_chan)
        main_chan_num=find(all(ismember(all_channels,main_chan),2));
    end
    if ~isnumeric(chan_nearby)
            num_nearby=find(all(ismember(all_channels,chan_nearby),2));
    end
n_lap_filters=length(main_chan_num);
main_chan_filtered=zeros(n_lap_filters,len_data_time);

        for filt=1:n_lap_filters %loop over the number of filters to apply
            % Main calculation of the Laplacian
           main_chan_filtered(filt,:)=data(main_chan_num(filt),:) - ...
               mean(data(num_nearby(filt,:))); 
        end
    data_aft(main_chan_num,:)=main_chan_filtered;
end