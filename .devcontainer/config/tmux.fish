function t
    tmux $argv
end

function ta
    if test (count $argv) -eq 0
        tmux a
    else if tmux has-session -t $argv[1] 2>/dev/null
        tmux a -t $argv[1]
    else
        printf "Session '%s' does not exist. Creating a new session.\n" $argv[1]
        sleep 1
        tmux new -s $argv[1]
    end
end

function tn
    tmux new -s $argv
end

function tls
    tmux ls
end
