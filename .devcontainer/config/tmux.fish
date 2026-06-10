function t
    tmux $argv
end

function ta
    if test (count $argv) -eq 0
        tmux a
    else
        tmux a -t $argv
    end
end

function tn
    tmux new -s $argv
end

function tls
    tmux ls
end
